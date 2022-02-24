scr_path = '要分类的图片目录' # 源目录
dst_path = '目标目录' # 目标目录

model = pdx.load_model('模型路径') # 加载模型
# print(model.get_model_info())# 显示信息

import glob

test_list = glob.glob(scr_path+'/*')
test_df = pd.DataFrame() # 创建表结构
# 将所有图片装进表中
for i in range(len(test_list)):
    img = Image.open(test_list[i]).convert('RGB')
    img = np.asarray(img, dtype='float32') # 转换数据类型

    result = model.predict(img[:, :, [2, 1, 0]]) # 预测结果
    test_df.at[i, 'name'] = str(test_list[i]).split('/')[-1] # 文件名
    test_df.at[i, 'cls'] = int(result[0]['category_id']) # 类别

print(test_df)

# 移动图片进行分类
for i in range(len(test_df)):
    img_path = os.path.join(scr_path, test_df.at[i, 'name']) # i 元素在列中的位置 ，name 列名
    save_path = os.path.join(dst_path, str(test_df.at[i, 'cls']))

    if not os.path.exists(save_path):
        os.makedirs(save_path)  # 没有则创建文件夹

    try:
        shutil.move(img_path, save_path)  # 移动图片到目标路径
    except Exception as e:
        print(e)  # 抛出错误信息
