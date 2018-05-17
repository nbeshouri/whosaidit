"""
This module contains the scrapers used to gather the data for the models.

Note:
    Scraped data is stored in `Munch` objects, which are just dictionaries
    whose values can be accessed using `munch.key`.

Todo:
    * `Scraper` class is too loosely defined and there's too much code
        duplication in the `scrape` method.
    * Lack of docstrings.

"""

from bs4 import BeautifulSoup
import requests
import re
from munch import Munch
import joblib
import os
import numpy as np
import time
from glob import glob
from . import utils


data_dir_path = os.path.join(os.path.dirname(__file__), 'data')
archived_data_dir_path = os.path.join(data_dir_path, 'archived')


class Scraper():
    
    def scrape(self):
        raise NotImplemented()
    
    @staticmethod
    def wait():
        delay = 0.1 + np.abs(np.random.normal(0, 1))
        time.sleep(delay)
    
    @staticmethod    
    def get_url_header(url):
        return re.split(r'(?<=\w)/(?=\w)', url)[0]


class BuffyScraper(Scraper):
    
    def scrape_ep_links(self, url):
        url_header = self.get_url_header(url)
        res = requests.get(url)
        soup = BeautifulSoup(res.text, 'lxml')
        ep_links = soup.find_all(name='a', href=re.compile(r'tran.html'))
        extracted_data = []
        for ep_link in ep_links:
            link_data = Munch(
                ep_title=ep_link.text,
                ep_code=ep_link['href'][:3],
                transcript_url=url + '/' + ep_link['href']  # Link relative to transcript page.
            )
            extracted_data.append(link_data)
        return extracted_data
        
    def scrape_episode(self, ep_data):
        res = requests.get(ep_data.transcript_url)
        soup = BeautifulSoup(res.text, 'lxml')
        big_string = '\n'.join(soup.stripped_strings)
        
        # Remove hard line breaks (hard wrapped lines appear to 
        # end in spaces, so I don't add one).
        big_string = re.sub(r'\n^\s{2,4}(?=\S)', '', big_string, flags=re.MULTILINE)
        # At least one ep (the last one) has these \r characters.
        big_string = re.sub(r'\r', '', big_string)
    
        # Seasons 1-6 uses "Speaker: Text" format while season 7
        # uses "SPEAKER\ntext", so we have to use different regex
        # patterns.
        # TODO: Handle two multi-part names (e.g. Obi-wan, 
        # Bob the Builder). There aren't any in the main cast, but still.
        if ep_data.ep_code < '123':
            pattern = r'^(\w+):\s+(\w.*)'
        else:
            pattern = r'^([A-Z]+)\n(\w.*)'
        matches = re.findall(pattern, big_string, flags=re.MULTILINE)
        dialogue_elements = []
        for i, (speaker, text) in enumerate(matches):
            dialogue_element = Munch(
                speaker=speaker.strip().capitalize(), 
                text=text.strip(), 
                location=i, 
                ep_code=ep_data.ep_code, 
                ep_title=ep_data.ep_title,
            )
            dialogue_elements.append(dialogue_element)
        
        return dialogue_elements
        
    def scrape(self):
        listing_url = 'http://www.buffyworld.com/buffy/transcripts/'
        file_name = 'buffy_dialogue_elements.pickle'
        ep_links = self.scrape_ep_links(listing_url)
        dialogue_elements = []
        for ep_data in ep_links[1:]:  # Don't scrap the pilot.
            # if ep_data.ep_code < '144':
            #     continue
            print(f'Scraping episode: {ep_data.ep_code} - {ep_data.ep_title}')
            dialogue_elements.extend(self.scrape_episode(ep_data))
            self.wait()
        utils.archive_data(file_name)    
        dump_path = os.path.join(data_dir_path, file_name)
        joblib.dump(dialogue_elements, dump_path)


class FuturamaScraper(Scraper):
    
    def scrape_episode(self, ep_data):
        """
        Return a list of bunch objects with the keys in the form:
        [character, season, episode, utterance, text]
        
        """
        res = requests.get(ep_data.transcript_url)
        soup = BeautifulSoup(res.text, 'lxml')
        speaker_tags = soup.find_all(name='div', class_='poem')
        dialogue_elements = []
        if len(speaker_tags) < 50:
            print(f'"{ep_data.ep_title}" has a funky format, skipping...')
            return dialogue_elements
        for i, speaker_tag in enumerate(speaker_tags):
            tag_strings = list(speaker_tag.p.strings)
            joined_string = ''.join(tag_strings)
            text_match = re.match(r'^.*?: (.*)', joined_string, re.DOTALL)
            speaker_name_tag = speaker_tag.find(name=re.compile('b|i'))
            
            # Sometimes narration elements get a poem class. These
            # are usually missing a speaker tag or a dialogue
            # section following a ': '.
            if text_match is None or speaker_name_tag is None:
                continue
            
            # Try to extract the speaker string.
            speaker = speaker_name_tag.string
            if speaker is None and speaker_name_tag.a is not None:
                speaker = speaker_name_tag.a.string
            
            # Continue if we still can't find the speaker.
            if speaker is None:
                continue

            speaker = speaker.strip()
            text = text_match.group(1).strip()
            
            dialogue_element = Munch(
                speaker=speaker, 
                text=text, 
                location=i, 
                ep_code=ep_data.ep_code, 
                ep_title=ep_data.ep_title,
            )
            dialogue_elements.append(dialogue_element)
        
        return dialogue_elements  

    def scrape_ep_links(self, url):
        """
        Return a list of `Munches` representing episode links.
        
        """
        res = requests.get(url)
        soup = BeautifulSoup(res.text, 'lxml')
        ep_links = soup.find_all(name='a', href=re.compile(r'/Transcript:'))
        extracted_data = []
        for ep_link in ep_links:
            tr_tag = ep_link.find_parent(name='tr')
            data_tag = tr_tag.find(name='td', string=re.compile(r'S\d.*'))
            # These will be None in movies.
            # TODO: Handle movies.
            if data_tag is None:
                continue
            url_header = self.get_url_header(url)
            full_link = url_header + ep_link['href']
            ep_title = ep_link['title'].replace('Transcript:', '')
            ep_title = ep_title.replace('_', ' ')
            link_data = Munch(
                ep_code=data_tag.string.strip(),
                transcript_url=full_link,
                ep_title=ep_title
            )
            extracted_data.append(link_data)
        return extracted_data    
        
    def scrape(self):
        listing_url = 'https://theinfosphere.org/Episode_Transcript_Listing'
        file_name = 'futurama_dialogue_elements.pickle'
        ep_links = self.scrape_ep_links(listing_url)
        dialogue_elements = []
        for ep_data in ep_links:
            print(f'Scraping episode: {ep_data.ep_code} - {ep_data.ep_title}')
            dialogue_elements.extend(self.scrape_episode(ep_data))
            self.wait()
        utils.archive_data(file_name)    
        dump_path = os.path.join(data_dir_path, file_name)
        joblib.dump(dialogue_elements, dump_path)
