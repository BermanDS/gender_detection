import re
import gc
import os
import sys
import json
import uuid
import base64
import hashlib
import logging
import requests
import itertools
import warnings
import numpy as np
import pandas as pd
import pyarrow as pa
from time import time, sleep
import pyarrow.parquet as pq
from dateutil.parser import parse
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from jsonschema import ErrorTree, Draft3Validator


format_times = {
    '%Y-%m-%d':False,
    '%d-%m-%Y':True, 
    '%Y%m%d':False, 
    '%d-%m-%y':True, 
    '%d-%b-%Y':True, 
    '%d/%m/%Y':True,
    '%d.%m.%Y':True,
    '%d%m%Y':True,
    '%Y-%m-%d %H:%M:%S':False,
    '%Y-%m-%d %H:%M:%S,%f':False,
}

#########_JSON processing procedures #############################################
###################################################################################

class NpEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)): 
            return None
            
        elif isinstance(obj, datetime):
            return obj.isoformat()
        #else:
        #    return super(NpEncoder, self).default(obj)
        return json.JSONEncoder.default(self, obj)


def to_json(dic, fname, enc='utf-16'):
    """
    saving dicts to json
    """
        
    try:
        with open(fname+'.json', 'w',encoding=enc) as fp:
            json.dump(dic, fp, ensure_ascii=False)
        #print('Saving:',fname,'OK!')
    except TypeError:
        dcc={str(k):v for k,v in dic.items()}
        try:
            with open(fname+'.json', 'w',encoding=enc) as fp:
                json.dump(dcc, fp, cls=NpEncoder)
        except:
            print('Problem with type:',fname)
    except:
        print('Problem with saving:',fname)


def json_extract_internal_levels(obj):
    """
    Recursively fetch values from nested JSON to single level.
    """
    
    dc = {}
    
    def extract(obj, dc, key):
        """
        Recursively search for values of key in JSON tree.
        """

        if isinstance(obj, dict):
            for k, v in obj.items():
                if key == '':
                    full_key = k.lower()
                else:
                    full_key = '_'.join(map(str, [key,k])).lower()
                
                if isinstance(v, dict):
                    extract(v, dc, full_key)
                elif isinstance(v, list):
                    dc[full_key] = []
                    for item in v:
                        if isinstance(item, dict):
                            extract(item, dc, full_key)
                        else:
                            dc[full_key].append(item)
                    if dc[full_key] == []:
                        del dc[full_key]
                else:
                    dc[full_key] = v if str(v).lower() not in ['none', 'nan','nat',''] else '0'
            
        elif isinstance(obj, list):
            dc[key] = []
            for item in obj:
                if isinstance(item, dict):
                    extract(item, dc, key)
                else:
                    dc[key].append(item)
            if dc[key] != []:
                del dc[key]
        
        else:
            dc[key] = obj if str(obj).lower() not in ['none', 'nan','nat', ''] else '0'
           
        return dc

    values = extract(obj, dc, '')

    return values


def json_validate(list_of_dc : list, schema : dict):
    """
    Check json to schema
    """
    
    errors = []
    try:
        validator = Draft3Validator(schema)
        errors = [i for i,x in enumerate(list_of_dc) if ErrorTree(validator.iter_errors(x))]
        
        return True, ','.join(map(str, errors))
    except Exception as err:
        return False, err        


######################################################################################################
######################### Parquet processing #########################################################
######################################################################################################


def append_to_parquet_table(dataframe, **pars):
    """Method writes/append dataframes in parquet format.

    This method is used to write pandas DataFrame as pyarrow Table in parquet format. If the methods is invoked
    with writer, it appends dataframe to the already written pyarrow table.

    :param dataframe: pd.DataFrame to be written in parquet format.
    :param filepath: target file location for parquet file.
    :param writer: ParquetWriter object to write pyarrow tables in parquet format.
    :return: ParquetWriter object. This can be passed in the subsequenct method calls to append DataFrame
        in the pyarrow Table
    """
    
    if dataframe.empty:
        return 'empty dataset', False
    
    if os.path.isfile(pars['filepath']) or os.path.exists(pars['filepath']):
        if pars['check_column'] != '':
            if pd.read_parquet(pars['filepath'], filters=[(pars['check_column'], '=', pars['check_value'])])[pars['check_column']].count() > 0:
                return 'no reason for appending data', False
        elif pars['solo']:
            dataframe = pd.read_parquet(pars['filepath'])\
                          .merge(dataframe, how = 'outer')\
                          .drop_duplicates()
    
    num_rows, chk_rows = len(dataframe), 0
    if pars['max_size'] > 0 and pars['max_size'] / num_rows < .95 and not pars['solo']:
        parts = np.int16(np.round(num_rows / pars['max_size']))
        for i in range(parts):
            
            if i == (parts - 1):
                if pars['schema']: table = pa.Table.from_pandas(dataframe.iloc[chk_rows : ], pars['schema'])
                else: table = pa.Table.from_pandas(dataframe)
            else:
                if pars['schema']: table = pa.Table.from_pandas(dataframe.iloc[chk_rows : chk_rows + pars['max_size']], pars['schema'])
                else: table = pa.Table.from_pandas(dataframe)
                
                chk_rows += pars['max_size']

            pq.write_to_dataset(table,\
                                root_path=pars['filepath'],\
                                use_deprecated_int96_timestamps = True,\
                                compression=pars['compression'])
    else:
        if pars['schema']: table = pa.Table.from_pandas(dataframe, pars['schema'])
        else: table = pa.Table.from_pandas(dataframe)
        
        if pars['solo']:
            writer = pq.ParquetWriter(pars['filepath'], table.schema)
            writer.write_table(table=table)
            writer.close()
        else:
            pq.write_to_dataset(table,\
                                root_path=pars['filepath'],\
                                use_deprecated_int96_timestamps = True,\
                                compression=pars['compression'])
    
    return '', True


def load_parquet_to_array(parquet_file, filter_set = [], out_columns = [], slice_count = 0, offset = 0, random_choice = False) -> list:
    """
    Downloading data from parquet table with filters
    return True/False (loading status), [data], 'error message'
    """
    
    if out_columns == []:
        return False, [], 'not defined output columns'
    elif random_choice:
        if filter_set != []:
            try:
                count_rows = pq.read_table(parquet_file,\
                                           filters = filter_set,\
                                           columns = out_columns).num_rows
                slice_count = np.min([count_rows, slice_count]) if slice_count > 0 else count_rows
                offset = np.max([np.min([offset, count_rows - slice_count]), 0])
                return True, pq.read_table(parquet_file,\
                                           filters = filter_set,\
                                           columns = out_columns)\
                               .slice(offset)\
                               .take(np.random.choice(list(range(offset, slice_count+offset)), slice_count))\
                               .to_pandas().values, ''
            except Exception as err:
                return False, [], err
        else:
            try:
                count_rows = pq.read_table(parquet_file,\
                                           columns = out_columns).num_rows
                slice_count = np.min([count_rows, slice_count]) if slice_count > 0 else count_rows
                offset = np.max([np.min([offset, count_rows - slice_count]), 0])
                return True, pq.read_table(parquet_file,\
                                           columns = out_columns)\
                               .slice(offset)\
                               .take(np.random.choice(list(range(offset, slice_count+offset)), slice_count))\
                               .to_pandas().values, ''
            except Exception as err:
                return False, [], err
    elif slice_count > 0:
        if filter_set != []:
            try:
                count_rows = pq.read_table(parquet_file,\
                                           filters = filter_set,\
                                           columns = out_columns).num_rows
                offset = np.max([np.min([offset, count_rows - slice_count]), 0])
                return True, pq.read_table(parquet_file,\
                                           filters = filter_set,\
                                           columns = out_columns)\
                               .slice(offset, slice_count)\
                               .to_pandas().values, ''
            except Exception as err:
                return False, [], err
        else:
            try:
                count_rows = pq.read_table(parquet_file,\
                                           filters = filter_set,\
                                           columns = out_columns).num_rows
                offset = np.max([np.min([offset, count_rows - slice_count]), 0])
                return True, pq.read_table(parquet_file,\
                                           columns = out_columns)\
                               .slice(offset, slice_count)\
                               .to_pandas().values, ''
            except Exception as err:
                return False, [], err
    else:
        if filter_set != []:
            try:
                return True, pq.read_table(parquet_file,\
                                           filters = filter_set,\
                                           columns = out_columns)\
                               .to_pandas().values, ''
            except Exception as err:
                return False, [], err
        else:
            try:
                return True, pq.read_table(parquet_file,\
                                           columns = out_columns)\
                               .to_pandas().values, ''
            except Exception as err:
                return False, [], err

####################################################################################
######## directories processing procedures ###########################################
####################################################################################

def get_size_of_directory(path = '.', output = 'Mb', print_result = True) -> float:
    """
    get size of definite directory in bytes (kbytes, Mbytes, Gbytes)
    """
    
    #nbytes = sum(d.stat().st_size for d in os.scandir(parquet_path) if d.is_file())
    
    nbytes = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            nbytes += os.path.getsize(os.path.join(root, file))
    
    if output == 'Mb':
        ndiv = 1000000
    elif output == 'Kb':
        ndiv = 1000
    elif output == 'Gb':
        ndiv = 1000000000
    else:
        ndiv = 1
    
    nbytes = round(nbytes/ndiv, 2)
    
    if print_result: print(f"Size of path: {path} is {nbytes} {output}!")

    return nbytes


def remove_files_in_directory(path = '', print_result = True) -> None:
    """
    remove files in definite directory
    """
    
    #nbytes = sum(d.stat().st_size for d in os.scandir(parquet_path) if d.is_file())
    nbytes = []
    nbytes.append(get_size_of_directory(path))
    
    for root, dirs, files in os.walk(path):
        for file in files:
            os.remove(os.path.join(root, file))
        
    nbytes.append(get_size_of_directory(path) * -1)
    
    if print_result: print(f"Removed in path: {path} {np.sum(nbytes)}!")


def get_path_for_name(root_path, search_name, is_dir = False) -> list:
    """
    Searching procedure in root_path area for definite search_name
    """
    
    result = []
    if root_path == '': return result
    
    for root, _, files in os.walk(root_path):
        if is_dir and search_name in root:
            result.append(root)
        elif not is_dir:
            tmp_list = [f for f in files if search_name in f]
            for f in tmp_list:
                result.append(os.path.join(root, f))
    
    return result


####################################################################################
######## cleansing processing procedures ###########################################
####################################################################################

def cleanhtml(raw_html):
    """
    clean raw html from tags and spaces
    """

    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    
    for i in ['\t','\n']:
        cleantext = cleantext.replace(i, ' ')
    cleantext = re.sub(' +', ' ', cleantext).strip()
    if cleantext in ['-','','\n'] or 'Warning:' in cleantext:return '0'
    else: return cleantext


def hascyrillic(text):
    """Checking of existance cyrillic words"""

    return bool(re.search('[\u0400-\u04FF]', str(text)))


def haseng(text):
    """Checking the existing the latin words in text"""
    
    lower = set('ABCDEFGHIJKLMNOPQRSTUVWXYZqwertyuiopasdfghjklzxcvbnm')
    
    return lower.intersection(text) != set()


def make_hash(*args):
    """
    procedure of hashing values in args
    """
    
    return hashlib.md5('-'.join(map(str, args)).encode()).hexdigest()


def make_guid(*args):
    """
    procedure of making GUID from values in args
    """
    
    return str(uuid.uuid5(uuid.NAMESPACE_OID, '-'.join(map(str, args))))

############################################################################################
########### logging ########################################################################
############################################################################################


FORMAT = '%(asctime)s - [%(levelname)s] : resource - %(location)s, %(message)s'

class CustomAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        my_context = kwargs.pop('kind', self.extra['kind'])
        print('Data - [%s]: %s' % (my_context, msg))
        return 'Data - [%s]: %s' % (my_context, msg), kwargs


class CustomFilter(logging.Filter):

    COLOR = {
        "DEBUG": "GREEN",
        "INFO": "GREEN",
        "WARNING": "YELLOW",
        "ERROR": "RED",
        "CRITICAL": "RED",
    }

    def filter(self, record):
        record.color = CustomFilter.COLOR[record.levelname]
        return True


class CustomLOG:
    
    def __init__(self, format_str: str, log_path: str, resource: str):
        
        self.formats = format_str
        self.log_path = log_path
        self.resource = resource
        self.not_use_log_level = ''
        self.levels = ['info', 'warning','error']
        self.msg_format = 'Data - [%s]: %s \n'
        self.pars = {
            'asctime':datetime.today().strftime('%Y-%m-%d %H:%M:%S'),
            'levelname':'INFO',
            'location':resource,
            'message':'',
        }
    

    def set_log_level(self, not_use_level = ''):
        

        self.levels = ['info', 'warning','error']

        if not_use_level in self.levels:
            self.not_use_log_level = not_use_level
            self.levels.remove(not_use_level)
        else:
            self.not_use_log_level = ''


    def info(self, msg, kind):
        
        self.pars['levelname'] = 'INFO'
        self.pars['message'] = self.msg_format % (kind, msg)
        self.pars['asctime'] = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        
        if 'info' in self.levels:
            with open(self.log_path, 'a') as f:
                info = self.formats%self.pars
                f.write(info)
    
    
    def warning(self, msg, kind):
        
        self.pars['levelname'] = 'WARNING'
        self.pars['message'] = self.msg_format % (kind, msg)
        self.pars['asctime'] = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        
        if 'warning' in self.levels:
            with open(self.log_path, 'a') as f:
                info = self.formats%self.pars
                f.write(info)
    

    def error(self, msg, kind):
        
        self.pars['levelname'] = 'ERROR'
        self.pars['message'] = self.msg_format % (kind, msg)    
        self.pars['asctime'] = datetime.today().strftime('%Y-%m-%d %H:%M:%S')

        if 'error' in self.levels:
            with open(self.log_path, 'a') as f:
                info = self.formats%self.pars
                f.write(info)
                
#########################################################################################
####FTP_connections #####################################################################
#########################################################################################

class FTPsource:
    """
    SMBprotocol connection class.
    """

    def __init__(self, dir_name, domain, username, password,\
                 start_date_of_files = '', temp_dir = '', end_date_of_files = '',\
                 port = 445, kind_ftp = 'smb'):
        
        self.dir = dir_name
        self.kind_connection = kind_ftp
        self.temp_dir = temp_dir
        self.domain = domain
        self.user = username
        self.password = password
        self.port = port
        self.start_fdate = start_date_of_files
        self.end_fdate = end_date_of_files
        self.df_temp = pd.DataFrame()
        self.files = {}
        self.cur = None
        self.conn = None
        self.logger = None
        self.stat_info = None

    
    def connect(self):
        """
        Connect to a Postgres database.
        """
        
        ### SMB server
        if self.conn is None and self.kind_connection == 'smb':
            self.cur = smbclient
            smbprotocol.logger.disabled = True
            smbprotocol.logging.disable(level = logging.DEBUG)
            self.cur.logger.disabled = True
            ###Connection to domain
            try:
                self.conn = Connection(uuid.uuid4(), self.domain, self.port)
                self.conn.connect(Dialects.SMB_3_0_2, timeout=3600)
                self.conn.require_signing = False
                self.conn.disconnect(Dialects.SMB_3_0_2)
                
                self.cur.ClientConfig(username=self.user, password=self.password)
                self.cur.reset_connection_cache()
                
            except Exception as error:
                if self.logger:
                    self.logger.error(f"Connection to FTP {self.domain}: {error}", kind = 'connection to FTP')
        ### FTP server
        elif self.conn is None and self.kind_connection == 'ftp':
            self.cur = ftplib.FTP()
            ###Connection to domain
            try:
                self.cur.connect(host = self.domain, port = self.port, timeout=60)
                self.cur.login(user=self.user, passwd=self.password)
                                
            except Exception as error:
                if self.logger:
                    self.logger.error(f"Connection to FTP {self.domain}: {error}", kind = 'connection to FTP')
            
        ###Connection to directory
        if self.stat_info is None and self.kind_connection == 'smb':
            try:
                self.stat_info = self.cur.lstat(self.dir)
                if self.logger:
                    self.logger.info(f"Connection to FTP directory {self.dir}: successfully!", kind = 'connection to FTP dir')
            except Exception as error:
                if self.logger:
                    self.logger.error(f"Connection to FTP directory {self.dir}: {error}", kind = 'connection to FTP dir')
        elif self.stat_info is None and self.kind_connection == 'ftp':
            try:
                self.stat_info = self.cur.getwelcome()
                
                if self.logger and self.dir in self.cur.nlst(self.dir):
                    self.logger.info(f"Connection to FTP directory {self.dir}: successfully!", kind = 'connection to FTP dir')
                elif self.logger:
                    self.logger.info("Connection to FTP established!", kind = 'connection to FTP dir')
            except Exception as error:
                if self.logger:
                    self.logger.error(f"Connection to FTP directory {self.dir}: {error}", kind = 'connection to FTP dir')
    

    def listdir(self):
        """
        List of files in directory
        """

        self.connect()
        self.files = {}
        ###Proccessing start date 
        if self.start_fdate == '' or check_dt_format(self.start_fdate) is None:
            self.start_fdate = datetime(1970, 1, 1)
        elif not isinstance(self.start_fdate, datetime):
            self.start_fdate = parse(self.start_fdate)
        
        ###Proccessing end date 
        if self.end_fdate == '' or check_dt_format(self.end_fdate) is None:
            self.end_fdate = datetime.today()
        elif not isinstance(self.end_fdate, datetime):
            self.end_fdate = parse(self.end_fdate)
        
        if self.end_fdate < self.start_fdate:
            self.end_fdate = datetime.today()
        
        if self.stat_info and self.kind_connection == 'smb':
            try:
                for x in self.cur.listdir(self.dir):
                    posname = pathlib.Path(os.path.join(self.dir, x))
                    postime = datetime.fromtimestamp(self.cur.lstat(posname).st_mtime)
                    if postime < self.start_fdate or postime > self.end_fdate: 
                        continue
                    self.files[posname] = {
                        'date':postime,
                        'name':posname.stem,
                    }
                    try:
                        self.cur.listdir(posname)
                        self.files[posname]['is_dir'] = True
                    except:
                        self.files[posname]['is_dir'] = False

                if self.logger:
                    self.logger.info(f"Defined {len(self.files.keys())} files in directory - {self.dir}.", kind = 'list of files in FTP dir')
            except Exception as error:
                if self.logger:
                    self.logger.error(f"Problem with file listing in FTP directory {self.dir}: {error}", kind = 'list of files in FTP dir')
                
        elif self.stat_info and self.kind_connection == 'ftp':
            try:
                files = self.cur.nlst(self.dir)
            except Exception as error:
                files = []
                if self.logger:
                    self.logger.error(f"Problem with file listing in FTP directory {self.dir}: {error}", kind = 'list of files in FTP dir')
            for name in files:
                self.files[name] = {
                     'name':name.split('/')[-1],
                }
                try:
                    postime = parse(self.cur.voidcmd("MDTM " + name)[4:].strip())
                    if postime < self.start_fdate or postime > self.end_fdate:
                        continue
                    self.files[name]['date'] = postime                    
                except:
                    self.files[name]['date'] = datetime(1970,1,1)
                
                try:
                    self.cur.size(name)
                    self.files[name]['is_dir'] = False
                except:
                    self.files[name]['is_dir'] = True
                
            if self.logger:
                self.logger.info(f"Defined {len(self.files.keys())} files in directory - {self.dir}.", kind = 'list of files in FTP dir')

        if self.files == {} and self.logger:
            self.logger.warning(f"Empty directory on FTP - {self.dir}", kind = 'list of files in FTP dir')
        self.close()

    
    def load_file_to_pandasDF(self, fname, enc = "utf8"):
        """
        loading file to the dataframe
        """
        
        self.connect()
        self.df_temp = pd.DataFrame()
                
        ###Reading file
        if self.kind_connection == 'smb':
            try:
                with self.cur.open_file(fname, encoding=enc) as f:
                    self.df_temp = pd.read_csv(f, engine='python', sep=';')
                if self.logger:
                    self.logger.info(f"Loaded {len(self.df_temp)} rows from csv - {fname}.", kind = 'dataframe downloading')
            except Exception as error:
                if self.logger:
                    self.logger.error(f"Problem with loading csv - {fname}: {error}", kind = 'dataframe downloading')
        
            if self.df_temp.empty and self.logger:
                self.logger.warning(f"Empty dataframe - {fname.stem}", kind = 'dataframe downloading')
            else:
                self.df_temp.rename({x:x.lower() for x in self.df_temp.columns}, axis = 1, inplace = True)
        elif self.kind_connection == 'ftp' and self.temp_dir != '':
            load = False
            try:
                resp = self.cur.retrbinary(f"RETR {self.dir}/" + fname,\
                                           open(os.path.join(self.temp_dir, fname), 'wb').write)
                if not resp.startswith('226 Transfer complete') and self.logger:
                    self.logger.error(f"Problem with loading - {fname} : {resp}", kind = 'dataframe downloading')
                else:
                    if fname.split('.')[-1].lower() in ['parquet']:
                        self.df_temp = pd.read_parquet(os.path.join(self.temp_dir, fname))
                        load = True
                    elif fname.split('.')[-1].lower() in ['csv','tsv','slk']:
                        self.df_temp = pd.read_csv(os.path.join(self.temp_dir, fname))
                        load = True
                    elif self.logger:
                        self.logger.error(f"Unknown file format - {fname}", kind = 'dataframe downloading')
                    
                    os.remove(os.path.join(self.temp_dir, fname))
            except Exception as error:
                if self.logger:
                    self.logger.error(f"Problem with loading - {fname}: {error}", kind = 'dataframe downloading')
            
            if load and self.logger and not self.df_temp.empty:
                self.logger.info(f"Loaded {len(self.df_temp)} rows from - {fname}.", kind = 'dataframe downloading')
                self.df_temp.rename({x:x.lower() for x in self.df_temp.columns}, axis = 1, inplace = True)
            elif load and self.logger:
                self.logger.warning(f"Empty dataframe - {fname}", kind = 'dataframe downloading')
        
        self.close()


    def upload_file(self, path, ftp_file = ''):
        """
        A function for uploading files to an FTP server
        @param self.cur: The file transfer protocol object
        @param path: The path to the file to upload
        """
        
        if not os.path.exists(path): return False
        success = False
        filename = path.split('/')[-1]
        if ftp_file == '':
            ftp_file = filename
        
        self.connect()
        if self.kind_connection == 'ftp' and self.dir != '' and self.cur:
            go_on = True
            # Test if directory exists. If not, create it
            if self.dir not in self.cur.nlst():
                try:
                    self.cur.mkd(self.dir)
                    if self.logger:
                        self.logger.info(f"Created directory: - {self.dir}.", kind = 'FTP make directory')
                except Exception as error: 
                    go_on = False                   
                    if self.logger:
                        self.logger.error(f"Problem with creating directory - {self.dir} : {error}", kind = 'FTP make directory')
            elif f"{self.dir}/{ftp_file}" in self.cur.nlst(self.dir):
                if self.logger:
                    self.logger.info(f"File {ftp_file}: already exist!", kind = 'dataframe uploading')
                success = True
                go_on = False 
            if go_on:
                # Check if file extension is text format
                ext = filename.split('.')[-1]
                try:
                    if ext.lower() in ["txt", "htm", "html"]:
                        resp = self.cur.storlines(f"STOR {self.dir}/" + ftp_file, open(path, "rb"))
                    else:
                        resp = self.cur.storbinary(f"STOR {self.dir}/" + ftp_file, open(path, "rb"), 4096)
                    
                    if not resp.startswith('226 Transfer complete') and self.logger:
                        self.logger.error(f"Problem with uploading - {ftp_file}: {resp}", kind = 'dataframe uploading')
                    else:
                        success = True
                        self.logger.info(f"Uploaded - {ftp_file}: successfully!", kind = 'dataframe uploading')
                except ftplib.all_errors as error:
                    if self.logger:
                        self.logger.error(f"FTP error - {ftp_file}: {error}", kind = 'dataframe uploading')
                except Exception as error:
                    if self.logger:
                        self.logger.error(f"Some error - {ftp_file}: {error}", kind = 'dataframe uploading')
        self.close()
        
        return success