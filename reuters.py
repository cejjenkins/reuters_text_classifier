import re
from lxml import etree, objectify
from datetime import datetime

class ReutersSGMLParser():
    """A helper class for parsing Reuters-21578 XGML file formats"""
    def __init__(self):
        self.bad_char_pattern = re.compile(r"&#\d*;")
        self.document_pattern = re.compile(r"<REUTERS.*?<\/REUTERS>", re.S)
        self.date_pattern = re.compile(r'[0-9]+-[A-Z]{3}-[0-9]{4} *[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]+')

    def empty_row(self):
        """Get an empty rows which can be transformed into a dataframe"""
        rows = {
            'old_id'     : [],
            'new_id'     : [],
            'has_topics' : [],
            'date'       : [],
            'topics'     : [],
            'places'     : [],
            'people'     : [],
            'orgs'       : [],
            'exchanges'  : [],
            'companies'  : [],
            'title'      : [],
            'dateline'   : [],
            'body'       : [],
            'author'     : [],
            'cgi_split'  : [],
            'lewis_split': []
        }
        return rows

    def get_text(self, elem, tagname, d_tag = False):
        """Get the text of a tag or empty string"""
        txt = getattr(elem, tagname, '')
        if txt == '':
            return ''
        if d_tag:
            txt = txt.D
        txt = txt.text.strip()
        return txt

    def get_date(self, elem, tagname):
        """Get the datetime of a tag or empty string"""
        date_str = getattr(elem, tagname, '')
        if date_str == '':
            return ''
        date_str = date_str.text.strip()
        try:
            date_str = self.date_pattern.findall(date_str)[0]
        except IndexError as ie:
            print('Cannot find date patter in: %s' % date_str)
            return ''
        date = datetime.strptime(date_str, '%d-%b-%Y %H:%M:%S.%f')
        return date

    def parse_header(self, rows, doc):
        """parse the header.
        e.g. <REUTERS TOPICS="YES" LEWISSPLIT="TRAIN" CGISPLIT="TRAINING-SET" OLDID="5544" NEWID="1">"""
        items = dict(doc.items())
        rows[   'old_id'  ].append(items.get('OLDID', ''))
        rows[   'new_id'  ].append(items.get('NEWID', ''))
        rows[ 'has_topics'].append(bool(items.get('TOPICS', '')))
        rows[ 'cgi_split' ].append(items.get('CGISPLIT', ''))
        rows['lewis_split'].append(items.get('LEWISSPLIT', ''))

    def parse_string(self, str):
        # remove bad characters
        xml_data = self.bad_char_pattern.sub('', str)
        # find documents
        documents = self.document_pattern.findall(xml_data)
        # parse document's elements
        rows = self.empty_row()
        for doc in documents:
            xml_doc = objectify.fromstring(doc)
            # parse attributes of the header
            self.parse_header(rows, xml_doc)
            # read DATE
            rows[  'date'  ].append(self.get_date(xml_doc, 'DATE'))
            # read TOPICS
            rows[  'topics'  ].append(self.get_text(xml_doc, 'TOPICS', True))
            # read PLACES
            rows[  'places'  ].append(self.get_text(xml_doc, 'PLACES', True))
            # read PEOPLE
            rows[ 'people'  ].append(self.get_text(xml_doc, 'PEOPLE', True))
            # read ORGS
            rows[ 'orgs'  ].append(self.get_text(xml_doc, 'ORGS', True))
            # read EXCHANGES
            rows[ 'exchanges'  ].append(self.get_text(xml_doc, 'EXCHANGES', True))
            # read COMPANIES
            rows[ 'companies'  ].append(self.get_text(xml_doc, 'COMPANIES', True))
            # read the TEXT tag
            text = xml_doc.TEXT
            rows[ 'title'  ].append(self.get_text(text, 'TITLE'))
            rows['dateline'].append(self.get_text(text, 'DATELINE'))
            rows[  'body'  ].append(self.get_text(text, 'BODY'))
            rows[  'author'  ].append(self.get_text(text, 'AUTHOR'))
        return rows

    def parse(self, path):
        """parse a file from the Reuters dataset
        """
        # open xml file
        xml_data = ''
        try:
            xml_data = open(path, 'r', encoding="utf-8").read()
        except UnicodeDecodeError as ude:
            print('Failed to read %s as utf-8' % path)
            lines = []
            for line in open(path, 'rb').readlines():
                line = line.decode('utf-8','ignore') #.encode("utf-8")
                lines.append(line)
            xml_data = '\n'.join(lines)
        return self.parse_string(xml_data)