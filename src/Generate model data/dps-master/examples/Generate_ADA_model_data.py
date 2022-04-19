#coding=utf-8

import os
import sys
import time

from absl import app
from absl import flags

import numpy as np
import pandas as pd
import random 
import data_process_service.utils as utils

FLAGS = flags.FLAGS

flags.DEFINE_string('name', 'taobao_preprocess', 'Test name.')
flags.DEFINE_bool('test', False, 'Whether in test mode.')
flags.DEFINE_string('load_path', './data/', 'Path to load file.')
flags.DEFINE_string('save_path', './data/examples', 'Path to save file.')


def filter_items_with_multiple_cids_taobao_ctr(flags_obj, record):

    start_time = time.time()
    item_cate = record[['iid', 'cid']].drop_duplicates().groupby('iid').count().reset_index().rename(columns={'cid': 'count'})
    items_with_single_cid = item_cate[item_cate['count'] == 1]['iid']

    id_filter = utils.FILTER.IDFilter(flags_obj, record)
    record = id_filter.filter(record, 'iid', items_with_single_cid)
    print('filter items with multiple_cids time: {:.2f} s'.format(time.time() - start_time))

    return record


def filter_behavior(flags_obj, record):

    start_time = time.time()
    click = pd.DataFrame({'behavior': ['pv']})
    id_filter = utils.FILTER.IDFilter(flags_obj, record)
    record = id_filter.filter(record, 'behavior', click)
    print('filter behavior time: {:.2f} s'.format(time.time() - start_time))

    return record


def process_taobao(flags_obj):
    # Load dataset
    record = utils.load_csv(flags_obj, 'example.txt', index_col=0)
    # Ten core filter_cf
    #record = utils.filter_cf(flags_obj, record, 20)
    # This is example, the code will randomly generate several numbers that are not repeated in the range
    #Random = random.sample(range(0,2586700),2323988) 
    #record =  record.iloc[random]
    record, user_reindex_map, item_reindex_map = utils.reindex_user_item(flags_obj, record)
    # Save reindex user item map json file
    utils.save_reindex_user_item_map(flags_obj, user_reindex_map, item_reindex_map)
    # Dataset split
    train_record,  test_record = utils.skew_split_v3(flags_obj, record, [0.6, 0.4])
    train_skew_record, val_record,test_skew_record = utils.skew_extract_v3(flags_obj, test_record, [0.25,0.25,0.50])
    train_blend_record = pd.concat([train_record,train_skew_record],axis=0)   
    #print(train_blend_record.head(10))
    # Save csv files
    #utils.save_csv_record(flags_obj, record, train_record, val_record, test_record)
    utils.save_csv_record(flags_obj, record, train_record, val_record, test_skew_record,train_skew_record=train_skew_record)
    
    # Dat analysis
    utils.report(flags_obj, record)
    coo_record, train_coo_record, val_coo_record, test_coo_record, train_skew_coo_record = utils.generate_coo(flags_obj, record, train_record, val_record, test_skew_record, train_skew_record=train_skew_record)
    _, _, _,_, train_blend_coo_record = utils.generate_coo(flags_obj, record, train_record, val_record, test_skew_record, train_skew_record=train_blend_record)
    utils.save_coo(flags_obj, coo_record, train_coo_record, val_coo_record, test_coo_record, train_skew_coo_record)
    utils.compute_popularity(flags_obj, train_coo_record)
    utils.compute_popularity(flags_obj, coo_record, 'popularity_all.npy')
    utils.compute_popularity(flags_obj, train_skew_coo_record,'popularity_skew.npy')
    utils.compute_popularity(flags_obj, train_blend_coo_record,'popularity_blend.npy')
    utils.generate_graph(flags_obj, train_coo_record)
    utils.generate_graph(flags_obj, train_skew_coo_record,'train_skew_coo_adj_graph.npz')
    utils.generate_graph(flags_obj, train_blend_coo_record,'train_blend_coo_adj_graph.npz')


def main(argv):

    flags_obj = flags.FLAGS

    process_taobao(flags_obj)


if __name__ == "__main__":

    app.run(main)

