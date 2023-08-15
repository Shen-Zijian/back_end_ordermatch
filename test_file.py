import gdown
import pandas as pd

# Google Drive 文件的链接
url = 'https://drive.google.com/file/d/1ECeJ35A2SzgRMREWM_BkcWQTN-kf1wAP/view?usp=drive_link'

# 将链接转换为直接下载链接
file_id = url.split('/')[-2]
d_url = 'https://drive.google.com/uc?id=' + file_id

# 下载文件
gdown.download(d_url, 'driver_info.csv', quiet=False)

# 读取 CSV 文件为 DataFrame
df = pd.read_csv('driver_info.csv')

# 查看 DataFrame
print(df)