import time

import requests
from fake_useragent import UserAgent

from lxml import html


class ProxiedRequest(object):
    """
    A class that provides an automatic way of obtaining a list of rotating proxies used to make web requests. 
    It scrapes proxies from us-proxy.com.
    """
    
    def __init__(self, proxy_test_url, proxies=[], connect_timeout=5, read_timeout=10, good_proxy_limit=25):
        self.proxies = proxies
        self.proxy_index = 0
        self.proxy_test_url = proxy_test_url
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.good_proxy_limit = good_proxy_limit
    
    
    def get(self, url, use_proxy=True):
        """
        Performs get request on the provided URL. It automatically adds a random user agent header and provides the option to add any proxy.
        """
        timeout = (self.connect_timeout, self.read_timeout)
        result = None

        ua = UserAgent().chrome
        proxies = None
        if use_proxy:
            if not self.proxies:
                self.proxies.extend(self._fetch_proxies())

            # round robin proxy choice
            proxy_choice = self.proxies[self.proxy_index]            
            proxies = self._proxies_for_proxy(proxy_choice)

            try:
                result = requests.get(url, headers={'User-Agent': ua}, proxies=proxies, timeout=timeout)
            except (requests.exceptions.ProxyError, requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout) as e:
                # remove bad proxy from the list
                print(self.proxy_index, len(self.proxies))
                self.proxies.pop(self.proxy_index)
                raise e
            
            # increment proxy index to perform round robin
            self.proxy_index += 1
            if self.proxy_index >= len(self.proxies):
                self.proxy_index = 0
        else:
            result = requests.get(url, headers={'User-Agent': ua}, timeout=timeout)

        return result

    
    def reset_proxies(self):
        """
        Empties the proxy list to forcefully refresh the available proxy list.
        """
        self.proxies = []
        self.proxy_index = 0
    

    def _proxies_for_proxy(self, proxy):
        proxy_url = 'http://' + proxy['ip'] + ':' + proxy['port']
        proxies = {'http': proxy_url}
        if proxy['https'] == 'yes':
            proxies['https'] = proxy_url

        return proxies


    def _test_proxy(self, proxy):
        timeout = (self.connect_timeout, self.read_timeout)
        proxies = self._proxies_for_proxy(proxy)

        good = False
        try:
            ua = UserAgent().chrome
            result = requests.get(self.proxy_test_url, headers={'User-Agent': ua}, proxies=proxies, timeout=timeout)
            if 'ERROR' not in result.text:
                good = True
        except (requests.exceptions.ProxyError, requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout) as e:
            pass

        return good


    def _fetch_proxies(self):
        """
        Fetches public proxy list from https://www.us-proxy.org/.
        """
        result = self.get("https://www.us-proxy.org/", use_proxy=False)
        tree = html.fromstring(result.text)
        proxies = []

        table_rows = tree.xpath("//table[@id='proxylisttable']/tbody/tr")
        good_count = 0
        for row in table_rows:
            columns = row.xpath("td")
            ip = columns[0].text
            port = columns[1].text
            code = columns[2].text
            country = columns[3].text
            anonymity = columns[4].text
            google = columns[5].text
            https = columns[6].text
            last_checked = columns[7].text

            if anonymity.lower() != 'transparent':
                proxy = {
                    'ip': ip,
                    'port': port,
                    'code': code,
                    'country': country,
                    'anonymity': anonymity,
                    'google': google,
                    'https': https,
                    'last_checked': last_checked
                }
                if self._test_proxy(proxy):                                
                    proxies.append(proxy)
                    good_count = good_count + 1

                    if good_count >= self.good_proxy_limit:
                        break

        return proxies
    
class RefreshingRequestor(object):
    """
    Wraps the ProxiedRequest object to automatically renew the proxy list after so many requests have been made.
    This ensures that proxies are not stale.
    """
    
    def __init__(self, proxy_test_url, refresh_at=100, sleep=None, good_proxy_limit=25):
        self.request_count = 0
        self.refresh_at = refresh_at
        self.sleep = sleep
        self.pr = ProxiedRequest(proxy_test_url, good_proxy_limit=good_proxy_limit)
        
        
    def get(self, url, use_proxy=True):
        data = self.pr.get(url, use_proxy=use_proxy)
        self.request_count += 1
        self._sleep_or_reset()
        return data
    
    
    def exhaustive_get(self, url, use_proxy=True, max_attempts=0):
        """
        Performs get request infinitely by default until a result is provided. 
        Denoted by max_attempts = 0.
        """
        data = None
        attempt_count = 0
        while data is None or not data.ok:
            try:
                data = self.get(url, use_proxy=use_proxy)
            except Exception as e:
                print(e)
                pass
            
            attempt_count += 1
                        
            if max_attempts > 0 and attempt_count >= max_attempts:                
                raise Exception('Exhausted attempts(' + str(max_attempts) + ') fetching: ' + url)
            
        return data
    
        
    def _sleep_or_reset(self):
        if self.request_count >= self.refresh_at:
            print("REFERSHING PROXIES")
            self.request_count = 0
            self.pr.reset_proxies()
        else:
            if isinstance(self.sleep, int):
                time.sleep(self.sleep)