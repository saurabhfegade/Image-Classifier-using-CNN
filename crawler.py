from icrawler.builtin import BaiduImageCrawler, BingImageCrawler, GoogleImageCrawler

google_crawler = GoogleImageCrawler(parser_threads=2, downloader_threads=4,
                                    storage={'root_dir': 'C:/Users/Saurabh/Desktop/'})
google_crawler.crawl(keyword='sandwich', offset=0, max_num=100,
                     date_min=None, date_max=None,
                     min_size=(200,200), max_size=None)



































