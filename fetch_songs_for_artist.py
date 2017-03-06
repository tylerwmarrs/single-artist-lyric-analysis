import argparse
import csv
import os
from urllib.parse import urlencode

from lyricanalysis.proxiedrequest import RefreshingRequestor, ProxiedRequest

from slugify import slugify
from lxml import html


def search_azlyrics_for_artist(requestor, artist):
    """
    Searches AZLyrics.com for specified artist.
    """
    artists = []
    base_url = 'http://search.azlyrics.com/search.php?'
    full_url = base_url + urlencode({'q': artist})
    
    result = requestor.exhaustive_get(full_url, max_attempts=5)
    tree = html.fromstring(result.text)
    
    # Check that there are artist results...
    search_panels = tree.xpath("//div[contains(@class, 'panel')]")
    if not search_panels:
        return artists
    
    artist_results_panel = None
    for panel in search_panels:
        label = panel.xpath(".//div[contains(@class, 'panel-heading')]")[0]
        label_text = label.text_content().lower()
        
        if 'artist' in label_text:
            artist_results_panel = panel
            break
    
    if artist_results_panel is None:
        return artists
       
    # fetch all artists and urls in results
    artist_results = artist_results_panel.xpath(".//td/a[@target='_blank']")
    for artist_result in artist_results:
        artist = artist_result.text_content().strip()
        url = artist_result.get('href')
        
        artists.append({
            'artist': artist,
            'url': url
        })
    
    return artists


def full_url_for_base_url(base_url):
    return base_url.replace('../', 'http://www.azlyrics.com/')


def scrape_artist_songs(requestor, artist, artist_page_url):
    """
    Scrapes songs from AZLyrics.com given an artist's URL. 
    """
    result = requestor.exhaustive_get(artist_page_url, max_attempts=5)
    tree = html.fromstring(result.text)
    
    songs = []
    album = None
    album_root = tree.xpath("//div[@id='listAlbum']/*[self::div[@class='album'] or self::a[@target='_blank']]")
    for child in album_root:
        
        # Parse the album from the page
        if child.tag == 'div':
            bold_tag = child.xpath("b")
            if bold_tag:
                album = bold_tag[0].text.replace('"', '')
            else:
                album = None
        elif child.tag == 'a':
            songs.append({
                'artist': artist,
                'album': album,
                'title': child.text,
                'lyrics_url': full_url_for_base_url(child.get('href'))
            })
            
    return songs


def scrape_lyrics(requestor, lyrics_url):
    """
    Scrape the lyrics page for the lyrics.
    """
    result = requestor.exhaustive_get(lyrics_url, max_attempts=20)
    html_text = result.text
    tree = html.fromstring(html_text)
    lyrics_html = tree.xpath("//div")
    
    lyrics = None
    try:
        lyrics = lyrics_html[21].text_content()
    except:
        pass
    
    return lyrics


def save_lyrics(save_dir, artist, title, lyrics):
    """
    Writes the lyrics file to the saved dir sluggifying the artist and title as the file name.
    """
    slugged = slugify(artist + ' ' + title, only_ascii=True) + '.txt'
    save_path = os.path.join(save_dir, slugged)
    
    try:
        f = open(save_path, 'w')
        f.write(lyrics)
    except:
        save_path = ''
    
    return save_path


def write_songs_csv(save_dir, songs):
    with open(os.path.join(save_dir, 'songs.csv'), 'w') as f:
        csv_writer = csv.writer(f, dialect='excel')
        csv_writer.writerow(['artist', 'album', 'title', 'lyrics_url', 'lyrics_file'])

        for song in songs:
            csv_writer.writerow([
                song['artist'],
                song['album'],
                song['title'],
                song['lyrics_url'],
                song['lyrics_file']
            ])
            

def write_songs_file(song_file, songs):
    csv_writer = csv.writer(song_file, dialect='excel')
    csv_writer.writerow(['artist', 'album', 'title', 'lyrics_url', 'lyrics_file'])

    for song in songs:
        csv_writer.writerow([
            song['artist'],
            song['album'],
            song['title'],
            song['lyrics_url'],
            song['lyrics_file']
        ])


def main():
    base_url = 'http://azlyrics.com'    
    requestor = RefreshingRequestor(base_url, good_proxy_limit=20, refresh_at=1000)
    
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("artist", help="Artist to search for")
    parser.add_argument("save_dir", help="Location to save to")
    args = parser.parse_args()
    
    artist = args.artist
    save_dir = os.path.join(args.save_dir, slugify(artist, only_ascii=True))
    results = search_azlyrics_for_artist(requestor, artist)
    if results:
        os.makedirs(save_dir, exist_ok=True)
        songs = scrape_artist_songs(requestor, artist, results[0]['url'])
        for song in songs:
            lyrics = scrape_lyrics(requestor, song['lyrics_url'])

            if lyrics:
                save_path = save_lyrics(save_dir, song['artist'], song['title'], lyrics)
                filename = save_path.split('/')[-1]
                song['lyrics_file'] = filename
                print("SAVED: " + song['title'])
            else:
                print("UNABLE TO FETCH: " + song['title'] + " FROM " + song['lyrics_url'])

        write_songs_csv(save_dir, songs)
    else:
        print("No results for: " + artist)
    

if __name__ == '__main__':
    main()