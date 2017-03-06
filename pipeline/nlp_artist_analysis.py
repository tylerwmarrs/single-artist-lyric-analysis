import os
import sys
import pickle
import re
import string

import luigi

from nltk.tokenize import word_tokenize
from nltk import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import matplotlib as mpl
mpl.use('Agg')

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

import pandas as pd
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)

if project_dir not in sys.path:
   sys.path.append(project_dir)

from fetch_songs_for_artist import *
from lyricanalysis.proxiedrequest import RefreshingRequestor
from lyricanalysis import corpus
from lyricanalysis import utils


class CreateDirectories(luigi.Task):
    output_dir = luigi.Parameter()
    
    def run(self):
        dirs = [
            os.path.join(self.output_dir, 'plots'),
            os.path.join(self.output_dir, 'data'),
            os.path.join(self.output_dir, 'topics')
        ]
        
        for d in dirs:
            os.makedirs(d, exist_ok=True)
            
        with self.output().open('w') as out_file:
            out_file.write('done')
            
    def output(self):
        return luigi.LocalTarget(os.path.join(self.output_dir, 'create_dirs.DONE'), is_tmp=True)


class ObtainArtistUrl(luigi.Task):
    artist = luigi.Parameter()
    output_dir = luigi.Parameter()
    
    def requires(self):
        return CreateDirectories(output_dir=self.output_dir)
    
    def run(self):
        requestor = RefreshingRequestor('http://azlyrics.com', refresh_at=25, sleep=None, good_proxy_limit=5)
        artists = search_azlyrics_for_artist(requestor, self.artist)
        
        if not artists:
            raise Exception("No artist URL found for %s" % self.artist)
            
        with self.output().open('w') as out_file:            
            out_file.write(artists[0]['artist'])
            out_file.write('\t')
            out_file.write(artists[0]['url'])
            out_file.write('\n')
    
    def output(self):
        output = os.path.join(self.output_dir, 'data', "artist_url.tsv")
        return luigi.LocalTarget(output)
    
    
class FetchArtistSongs(luigi.Task):   
    artist = luigi.Parameter()
    output_dir = luigi.Parameter()
    
    def requires(self):
        return ObtainArtistUrl(artist=self.artist, output_dir=self.output_dir)
    
    
    def run(self):
        requestor = RefreshingRequestor('http://azlyrics.com', refresh_at=25, sleep=None, good_proxy_limit=5)
        
        artist = None
        artist_url = None
        with self.input().open('r') as in_file:
            artist, artist_url = in_file.read().strip().split('\t')
        
        artist_songs = scrape_artist_songs(requestor, artist, artist_url)

        if not artist_songs:
            raise Exception("No artist songs found for %s %s" % (artist, artist_url))
            
        with self.output().open('w') as out_file:            
            for song in artist_songs:
                song_album = song['album']
                if song_album is None:
                    song_album = ''
                
                out_file.write('\t'.join([
                    song['artist'],
                    song_album,
                    song['title'],
                    song['lyrics_url']
                ]))
                
                out_file.write('\n')
    

    def output(self):
        output = os.path.join(self.output_dir, 'data', "artist_songs.tsv")
        return luigi.LocalTarget(output)

    
class FetchLyrics(luigi.Task):
    artist = luigi.Parameter()
    output_dir = luigi.Parameter()
    
    def requires(self):
        return FetchArtistSongs(artist=self.artist, output_dir=self.output_dir)
    
    def run(self):
        
        save_dir = os.path.join(self.output_dir, 'data', 'songs')
        os.makedirs(save_dir, exist_ok=True)
        
        songs = []
        with self.input().open('r') as in_file:
            for line in in_file:
                data = line.strip().split('\t')                
                songs.append({
                    'artist': data[0],
                    'album': data[1],
                    'title': data[2],
                    'lyrics_url': data[3]
                })
                
        if not songs:
            raise Exception("No songs to fetch (empty file).")
            
        requestor = RefreshingRequestor('http://azlyrics.com', refresh_at=300, sleep=None, good_proxy_limit=25)
        
        for song in songs:
            lyrics = scrape_lyrics(requestor, song['lyrics_url'])
            save_path = save_lyrics(save_dir, song['artist'], song['title'], lyrics)
            filename = save_path.split('/')[-1]
            song['lyrics_file'] = filename
        
        with self.output().open('w') as out_file:
            write_songs_file(out_file, songs)
    
    def output(self):
        output = os.path.join(self.output_dir, 'data', 'songs', 'songs.tsv')
        return luigi.LocalTarget(output)    
    
    
class CleanLyrics(luigi.Task):        
    artist = luigi.Parameter()
    output_dir = luigi.Parameter()
    
    tagged = re.compile('\[.*\]')
    produced = re.compile('[p|P]roduced.*')

    def _clean_lyrics(self, lyrics):
        new_lyrics = self.tagged.sub('', lyrics)
        new_lyrics = self.produced.sub('', new_lyrics)

        return new_lyrics
   
    def requires(self):
        return FetchLyrics(artist=self.artist, output_dir=self.output_dir)
    
    def run(self):
        song_dir = os.path.dirname(self.input().path)
        songs = corpus.load_songs(song_dir)
        
        for song in songs:
            song['lyrics'] = self._clean_lyrics(song['lyrics'])
            
        with self.output().open('wb') as out_file:
            pickle.dump(songs, out_file)
    
    def output(self):
        output = os.path.join(self.output_dir, 'data', 'clean_lyrics.pickle')
        return luigi.LocalTarget(
            output, 
            format=luigi.format.Nop
        )
    

class WordTokenizeAndStemLyrics(luigi.Task):    
    artist = luigi.Parameter()
    output_dir = luigi.Parameter()
    
    def _clean_tokens(self, text):
        return utils.stem_words(
            utils.remove_stop_words(
                corpus.stop_words(),
                word_tokenize(
                    utils.remove_punctuation(text)
                )
            )
        )
    
    def requires(self):
        return CleanLyrics(artist=self.artist, output_dir=self.output_dir)
    
    def run(self):
        with self.input().open('rb') as in_file:
            songs = pickle.load(in_file)
        
        for song in songs:
            song['word_tokens'] = self._clean_tokens(song['lyrics'])
            song['word_count'] = len(song['word_tokens'])
            
        with self.output().open('wb') as out_file:
            pickle.dump(songs, out_file)
            
    
    def output(self):
        output = os.path.join(self.output_dir, 'data', 'tokened_stemmed_lyrics.pickle')
        return luigi.LocalTarget(
            output, 
            format=luigi.format.Nop
        )

    
class MeasureWordFrequency(luigi.Task):
    artist = luigi.Parameter()
    output_dir = luigi.Parameter()
    
    def requires(self):
        return WordTokenizeAndStemLyrics(artist=self.artist, output_dir=self.output_dir)
    
    def run(self):
        with self.input().open('rb') as in_file:
            songs = pickle.load(in_file)
        
        normalized_word_frequencies = {}

        for song in songs:
            dist = FreqDist(song['word_tokens'])

            for w in dist:
                if not w in normalized_word_frequencies:
                    normalized_word_frequencies[w] = 0

                normalized_word_frequencies[w] += dist.freq(w)

        for w, v in normalized_word_frequencies.items():
            normalized_word_frequencies[w] = v / len(songs)
            
        df = pd.DataFrame.from_dict(normalized_word_frequencies, orient='index')
        title = 'Normalized Word Frequency\n%s' % (self.artist)
        word_freq = df.nlargest(25, 0).plot(
            kind='bar', 
            title=title, 
            legend=False
        )
        word_freq.set_xlabel("Word")
        word_freq.set_ylabel("Distribution")
        
        with self.output().open('wb') as out_file:
            word_freq.get_figure().savefig(out_file, dpi='figure')
    
    def output(self):
        output = os.path.join(self.output_dir, 'plots', 'word_frequency_plot.png')
        return luigi.LocalTarget(
            output, 
            format=luigi.format.Nop
        )
    
    
class SwearWordFrequency(luigi.Task):    
    artist = luigi.Parameter()
    output_dir = luigi.Parameter()
           
    def requires(self):
        return WordTokenizeAndStemLyrics(artist=self.artist, output_dir=self.output_dir)
    
    def run(self):
        swear_words = set(utils.stem_words(corpus.swear_words()))
        
        with self.input().open('rb') as in_file:
            songs = pickle.load(in_file)
        
        normalized_word_frequencies = {}

        for song in songs:
            dist = FreqDist(song['word_tokens'])

            for sw in swear_words:
                if not sw in normalized_word_frequencies:
                    normalized_word_frequencies[sw] = 0

                normalized_word_frequencies[sw] += dist.freq(sw)

        for w, v in normalized_word_frequencies.items():
            normalized_word_frequencies[w] = v / len(songs)
            
        df = pd.DataFrame.from_dict(normalized_word_frequencies, orient='index')
        title = 'Swear Word Frequency\n%s' % (self.artist)
        word_freq = df.nlargest(5, 0).plot(
            kind='bar', 
            title=title, 
            legend=False
        )
        word_freq.set_xlabel("Swear Word")
        word_freq.set_ylabel("Distribution")
        
        with self.output().open('wb') as out_file:
            word_freq.get_figure().savefig(out_file, dpi='figure')
    
    def output(self):
        output = os.path.join(self.output_dir, 'plots', 'swearword_frequency_plot.png')
        return luigi.LocalTarget(
            output, 
            format=luigi.format.Nop
        )
    

class SentimentAnalysis(luigi.Task):    
    artist = luigi.Parameter()
    output_dir = luigi.Parameter()

    def _sentiment_for_song(self, text):
        sid = SentimentIntensityAnalyzer()

        # can't use sent_tokenize with lyrics... just split on newline
        sentences = utils.split_sentences(text)
        total_ss = {
            'negative': 0,
            'positive': 0,
            'neutral': 0,
            'compound': 0
        }

        for sentence in sentences:
            ss = sid.polarity_scores(sentence)
            total_ss['negative'] = total_ss['negative'] + ss['neg']
            total_ss['positive'] = total_ss['positive'] + ss['pos']
            total_ss['neutral'] = total_ss['neutral'] + ss['neu']
            total_ss['compound'] = total_ss['compound'] + ss['compound']

        for key in total_ss:
            if len(sentences) > 0:
                total_ss[key] = total_ss[key] / len(sentences)

        return total_ss
    
    def requires(self):
        return CleanLyrics(artist=self.artist, output_dir=self.output_dir)
    
    def run(self):
        with self.input().open('rb') as in_file:
            songs = pickle.load(in_file)
            
        all_sentiments = {}
        for song in songs:
            sentiments = self._sentiment_for_song(song['lyrics'])

            # add sentiment to song data
            song['positive_sentiment'] = sentiments['positive']
            song['negative_sentiment'] = sentiments['negative']
            song['neutral_sentiment'] = sentiments['neutral']

            for key in sentiments:
                if key not in all_sentiments:
                    all_sentiments[key] = 0

                all_sentiments[key] = all_sentiments[key] + sentiments[key]

        # normalize for number of songs
        for key in all_sentiments:
            all_sentiments[key] = all_sentiments[key] / len(songs)
        
        # plot and save figure
        all_sentiments.pop('compound', 0)
        df = pd.DataFrame.from_dict(all_sentiments, orient='index')
        title = 'Sentiment Distribution\n%s' % (self.artist)
        sent_plot = df.plot(
            kind='bar', 
            title=title,
            legend=False
        )
        sent_plot.set_xlabel("Sentiment")
        sent_plot.set_ylabel("Distribution")
        
        with self.output()['plot'].open('wb') as out_file:
            sent_plot.get_figure().savefig(out_file, dpi='figure')
            
        # save the sentiments
        with self.output()['pickled'].open('wb') as out_file:
            pickle.dump(songs, out_file)
    
    def output(self):
        plot_file = os.path.join(self.output_dir, 'plots', 'sentiment_plot.png')
        plot = luigi.LocalTarget(plot_file, format=luigi.format.Nop)
        
        pickled_file = os.path.join(self.output_dir, 'data', 'song_sentiments.pickle')
        pickled = luigi.LocalTarget(pickled_file, format=luigi.format.Nop)
        return {
            'plot': plot,
            'pickled': pickled
        }
    

class RepetitivenessAnalysis(luigi.Task):    
    artist = luigi.Parameter()
    output_dir = luigi.Parameter()
    
    def requires(self):
        return CleanLyrics(artist=self.artist, output_dir=self.output_dir)
    
    def run(self):
        with self.input().open('rb') as in_file:
            songs = pickle.load(in_file)

        for song in songs:
            song['repetitiveness'] = utils.song_repetiveness(song['lyrics'], rate=2)
    
        # plot and save figure
        songs_df = pd.DataFrame.from_dict(songs)
        plot = sns.boxplot(songs_df.repetitiveness)
        
        title = 'Song Repetitiveness\n%s' % (self.artist)
        plot.set_title(title)
        
        with self.output()['plot'].open('wb') as out_file:
            plot.get_figure().savefig(out_file, dpi='figure')
            
        # save the sentiments
        with self.output()['pickled'].open('wb') as out_file:
            pickle.dump(songs, out_file)
    
    def output(self):
        plot_file = os.path.join(self.output_dir, 'plots', 'repetitiveness_plot.png')
        plot = luigi.LocalTarget(plot_file, format=luigi.format.Nop)
        
        pickled_file = os.path.join(self.output_dir, 'data', 'song_repetitiveness.pickle')
        pickled = luigi.LocalTarget(pickled_file, format=luigi.format.Nop)
        return {
            'plot': plot,
            'pickled': pickled
        }
    
    
class SongStatistics(luigi.Task):    
    artist = luigi.Parameter()
    output_dir = luigi.Parameter()
    
    def requires(self):
        return {
            'sentiment': SentimentAnalysis(artist=self.artist, output_dir=self.output_dir),
            'repetitiveness': RepetitivenessAnalysis(artist=self.artist, output_dir=self.output_dir)
        }
    
    def run(self):
        # load up the sentiment and repeitiveness data
        with self.input()['sentiment']['pickled'].open('rb') as in_file:
            sent_dict = pickle.load(in_file)
            
        with self.input()['repetitiveness']['pickled'].open('rb') as in_file:
            rep_dict = pickle.load(in_file)
                    
        sent_df = pd.DataFrame.from_dict(sent_dict)
        rep_df = pd.DataFrame.from_dict(rep_dict)
        
        songs_df = pd.merge(rep_df, sent_df, how='inner', on=['album', 'artist', 'title'])
        title = 'Song Statistics\n%s' % (self.artist)
        cols = ['title', 'repetitiveness', 'positive_sentiment', 'negative_sentiment']
        plot = songs_df[cols].plot(kind='bar', x='title', title=title)

        with self.output().open('wb') as out_file:
            plot.get_figure().savefig(out_file, dpi='figure')
    
    def output(self):
        output = os.path.join(self.output_dir, 'plots', 'song_statistics_plot.png')
        return luigi.LocalTarget(output, format=luigi.format.Nop)
    
    
class AlbumStatistics(luigi.Task):    
    artist = luigi.Parameter()
    output_dir = luigi.Parameter()
    
    def requires(self):
        return {
            'sentiment': SentimentAnalysis(artist=self.artist, output_dir=self.output_dir),
            'repetitiveness': RepetitivenessAnalysis(artist=self.artist, output_dir=self.output_dir)
        }
    
    def run(self):
        # load up the sentiment and repeitiveness data
        with self.input()['sentiment']['pickled'].open('rb') as in_file:
            sent_dict = pickle.load(in_file)
            
        with self.input()['repetitiveness']['pickled'].open('rb') as in_file:
            rep_dict = pickle.load(in_file)
                    
        sent_df = pd.DataFrame.from_dict(sent_dict)
        rep_df = pd.DataFrame.from_dict(rep_dict)
        
        songs_df = pd.merge(rep_df, sent_df, how='inner', on=['album', 'artist', 'title'])

        albums_groups = songs_df.groupby('album')[
            'repetitiveness', 
            'negative_sentiment', 
            'positive_sentiment'
        ].agg(['count', 'sum'])
        
        albums_groups['repetitiveness_normalized'] = albums_groups['repetitiveness']['sum'] / albums_groups['repetitiveness']['count']
        albums_groups['negative_sentiment_normalized'] = albums_groups['negative_sentiment']['sum'] / albums_groups['negative_sentiment']['count']
        albums_groups['positive_sentiment_normalized'] = albums_groups['positive_sentiment']['sum'] / albums_groups['positive_sentiment']['count']
        albums_groups.drop('repetitiveness', axis=1, inplace=True)
        albums_groups.drop('positive_sentiment', axis=1, inplace=True)
        albums_groups.drop('negative_sentiment', axis=1, inplace=True)
        album_stats = albums_groups.reset_index().stack()
        album_stats = album_stats.rename(columns = {
            'repetitiveness_normalized':'Repetitiveness', 
            'negative_sentiment_normalized': 'Negative Sentiment', 
            'positive_sentiment_normalized': 'Positive Sentiment', 
            'words_per_song': 'Avg. Words Per Song'
        })
        cols = ['album', 'Negative Sentiment', 'Positive Sentiment', 'Repetitiveness']
        title = 'Album Statistics\n%s' % (self.artist)
        plot = album_stats[cols].plot(kind='bar', x='album', title=title)
        
        with self.output().open('wb') as out_file:
            plot.get_figure().savefig(out_file, dpi='figure')
    
    def output(self):
        output = os.path.join(self.output_dir, 'plots', 'album_statistics_plot.png')
        return luigi.LocalTarget(output, format=luigi.format.Nop)

    
class TopicModeling(luigi.Task):    
    artist = luigi.Parameter()
    output_dir = luigi.Parameter()
    
    def _get_topics(self, model, feature_names, no_top_words):
        topics = []
        for topic_idx, topic in enumerate(model.components_):
            topic = " ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
            topics.append(topic)
            
        return "\n".join(topics)
    
    def requires(self):
        return CleanLyrics(artist=self.artist, output_dir=self.output_dir)
    
    def run(self):
        with self.input().open('rb') as in_file:
            songs = pickle.load(in_file)
        
        tf = TfidfVectorizer(
            analyzer='word', 
            ngram_range=(1,2), 
            min_df = 0, 
            stop_words = corpus.stop_words()
        )
        
        song_corpus = [song['lyrics'] for song in songs]
        tfidf_matrix = tf.fit_transform(song_corpus)
        feature_names = tf.get_feature_names()

        no_topics = 5

        # Run NMF
        nmf = NMF(
            n_components=no_topics, 
            random_state=1, 
            alpha=.1, 
            l1_ratio=.5, 
            init='nndsvd'
        ).fit(tfidf_matrix)

        # Run LDA
        lda = LatentDirichletAllocation(
            n_topics=no_topics, 
            max_iter=20, 
            learning_method='online', 
            learning_offset=50.,
            random_state=0
        ).fit(tfidf_matrix)

        no_top_words = 5
        
        # write topics per algorithm
        with self.output()['nmf'].open('w') as out_file:
            topics = self._get_topics(nmf, feature_names, no_top_words)
            out_file.write(topics)
            
        with self.output()['lda'].open('w') as out_file:
            topics = self._get_topics(lda, feature_names, no_top_words)
            out_file.write(topics)
        
    def output(self):
        nmf_file = os.path.join(self.output_dir, 'topics', 'nmf_topics.txt')
        lda_file = os.path.join(self.output_dir, 'topics', 'lda_topics.txt')
        return {
            'nmf': luigi.LocalTarget(nmf_file),
            'lda': luigi.LocalTarget(lda_file)
        }

    
class AllReports(luigi.WrapperTask):
    
    artist = luigi.Parameter()
    output_dir = luigi.Parameter()
    
    def requires(self):
        yield MeasureWordFrequency(artist=self.artist, output_dir=self.output_dir)
        yield SwearWordFrequency(artist=self.artist, output_dir=self.output_dir)
        yield SentimentAnalysis(artist=self.artist, output_dir=self.output_dir)
        yield RepetitivenessAnalysis(artist=self.artist, output_dir=self.output_dir)
        yield SongStatistics(artist=self.artist, output_dir=self.output_dir)
        yield AlbumStatistics(artist=self.artist, output_dir=self.output_dir)
        yield TopicModeling(artist=self.artist, output_dir=self.output_dir)
    

def main():
    luigi.run()


if __name__ == '__main__':
    main()