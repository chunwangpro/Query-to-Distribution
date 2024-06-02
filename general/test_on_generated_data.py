from schema import Query
from schema import Schema
from schema import Column
import datetime
import random

cardinality = 13364709
is_main = Column('is_main', 0, 1, int)
biz_order_id = Column('biz_order_id', 31635, 1146860131006968630, int)
pay_status = Column('pay_status', 1, 12, int)
is_detail = Column('is_detail', 0, 1, int)
auction_id = Column('auction_id', 0, 220463047172604086, int)
biz_type = Column('biz_type', 100, 52001, int)
buyer_flag = Column('buyer_flag', 0, 205, int)
options = Column('options', 0, 4611686022722355200, int)
buyer_id = Column('buyer_id', 21006, 2208724496142, int)
seller_id = Column('seller_id', 73, 2208694966044, int)
attribute4 = Column('attribute4', 0, 2, int)
logistics_status = Column('logistics_status', 1, 8, int)
status = Column('status', 0, 1, int)
s_time = datetime.datetime.strptime('2007-01-26 22:49:39', '%Y-%m-%d %H:%M:%S').timestamp()
e_time = datetime.datetime.strptime('2020-09-01 17:18:58', '%Y-%m-%d %H:%M:%S').timestamp()
gmt_create = Column('gmt_create', s_time, e_time, datetime)
s_time = datetime.datetime.strptime('2007-01-26 22:49:39', '%Y-%m-%d %H:%M:%S').timestamp()
e_time = datetime.datetime.strptime('2020-09-01 17:18:58', '%Y-%m-%d %H:%M:%S').timestamp()
end_time = Column('end_time', s_time, e_time, datetime)
s_time = datetime.datetime.strptime('2008-10-04 17:50:50', '%Y-%m-%d %H:%M:%S').timestamp()
e_time = datetime.datetime.strptime('2020-07-30 17:18:59', '%Y-%m-%d %H:%M:%S').timestamp()
pay_time = Column('pay_time', s_time, e_time, datetime)
from_group = Column('from_group', 0, 4, int)
sub_biz_type = Column('sub_biz_type', 0, 5007, int)
attributes = Column('attributes', '', '', str)
buyer_rate_status = Column('buyer_rate_status', 4, 7,int)
parent_id = Column('parent_id', 0, 1146860131006968630, int)
refund_status = Column('refund_status', 0, 14, int)
# columns = [biz_order_id, is_detail, seller_id, auction_id, biz_type, pay_status, options, buyer_id, status, gmt_create, from_group]
columns = [biz_order_id, refund_status, gmt_create, parent_id, sub_biz_type, is_detail, seller_id, auction_id, biz_type, pay_status, is_main, status, from_group, buyer_id, buyer_flag]
# columns = [biz_order_id, is_detail, auction_id, pay_status, options, buyer_id, from_group]
# columns = [biz_order_id]
table = Schema(columns, cardinality)
queries = []
types = {}
for col in columns:
    types[col.name] = col.type
# sql = "SELECT count(*) FROM tc_biz_order_0526 AS tc_biz_order WHERE is_main = 1 AND biz_type IN (5000, 1110, 6001, 8001, 760, 9999, 6868, 3600, 2100, 3000, 2700, 2600, 1400, 1410, 6800, 2500, 150, 3800, 3300, 3500, 2000, 110, 1102, 10000, 2410, 2400, 1500, 1201, 1200, 900, 620, 610, 600, 710, 500, 300, 200, 100) AND (options & 134217728 <> 134217728 OR options & 268435456 <> 268435456) AND buyer_id = 3893680462 AND options & 72057594037927936 <> 72057594037927936 AND options & 4503599627370496 <> 4503599627370496 AND options & 34359738368 <> 34359738368 AND options & 281474976710656 <> 281474976710656 AND options & 68719476736 <> 68719476736 AND options & 1073741824 <> 1073741824 AND status = 0 AND buyer_flag IN (5, 4, 3, 2, 1, 0) AND from_group = 0 AND attributes NOT LIKE '%;tbpwBizType:c2b2c;%' AND IFNULL(attribute4, 0) <> 1 AND IFNULL(attribute4, 0) <> 2"
# queries.append(Query(sql, types, 3))
with open('/data1/jisun.sj/query2distribute/query2distribute/datasets/buyers_0032.tc_biz_order_0526.dedup.card.in3', 'r') as f:
    lines = f.readlines()
    random.shuffle(lines)
    for query in lines:
        if int(query.split(',')[-1]) > 0:
            queries.append(Query(','.join(query.split(',')[:-1]), types, int(query.split(',')[-1])))

data_by_col = dict([(c.name, []) for c in columns])
with open('generated_dataset.csv', 'r') as f:
    for idx, line in enumerate(f.readlines()[1:]):
        vals = line[:-1].split(',')
        try:
            for col_id in range(len(columns)):
                data_by_col[columns[col_id].name].append(vals[col_id])
        except:
            print (idx, len(vals))
            raise Exception('haha')
qerrors = []
for query in queries:
    print (query.query)
    estimate = query.valid_rows(data_by_col).sum() * 2 + 1
    truth = query.true_cardinality
    if estimate < truth:
        qerrors.append(truth / estimate)
    else:
        qerrors.append(estimate / truth)
    print (truth, estimate, qerrors[-1])
import numpy as np
np.save('qerrors.npy', np.array(qerrors))
