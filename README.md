# dl_final2
第二個方法是Rapidfuzz 的文字比對架構
這個方法它不使用神經網路抓特徵，是用文字模糊匹配（Fuzzy String Matching）的方法。
作法先從影像URL中整理出檔案名稱當作關鍵字特徵，接著使用token_set_ratio演算法計算關鍵字與所有候選圖文之間的文字相似度，評估就會拿相似度最高的前五筆作為預測結果。

requirements.txt
放相關需要的套件

test_caption_list.csv
test.tsv
因為這份專案他沒有CNN是直接拿測試及資料比對的，因此只需要這兩份資料即可。
