import requests
import pdb

#url = 'https://www.k-auction.com/Auction/Premium/156?page_size=10&page_type=P&auc_kind=2&auc_num=156&page=18'
url = 'https://www.k-auction.com/api/Auction/2/197'

with open('art_list.tsv', 'w') as writer:
    writer.write('name\tprice_high\tprice_low\n')
    #for page in range(1, 17):
    for page in range(1, 5):
        #pdb.set_trace()
        res = requests.post(url, 
                            json={'auc_kind': '2', 'auc_num': 156, 
                                  'page': page, 'page_size': 10, 'page_type': 'P'}).json()

        for item in res['data']:
            artist_name = item['artist_name']
            price_low = item['price_estimated_low']
            price_high = item['price_estimated_high']
            
            print('artist', item['artist_name'], 
                  'price low', item['price_estimated_low'],
                  'price high', item['price_estimated_high'])
            # writer.write(f'{artist_name}\t{price_high}\t{price_low}\n')

pdb.set_trace()

