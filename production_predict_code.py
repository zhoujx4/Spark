    # coding:utf-8

import datetime
import pandas as pd
import pyspark.sql.functions as f
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql import types as t
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import pandas_udf, PandasUDFType
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.feature import MinMaxScalerModel, VectorAssembler


def load_data():

    ods_mk_xcx_log_action = spark.sql(
        'select * from bgy_data_platform.ods_mk_xcx_log_action') \
        .withColumn('date', f.to_date(f.col('channelid'))) \
        .withColumn('channelid', f.to_timestamp(f.col('channelid'))) \
        .cache()
    ods_mk_xcx_arean = spark.sql(
        'select * from bgy_data_platform.ods_mk_xcx_arean').cache()
    ods_mk_xcx_card_businesscard_c = spark.sql(
        'select * from bgy_data_platform.ods_mk_xcx_card_businesscard_c').cache()

    return ods_mk_xcx_log_action, ods_mk_xcx_arean, ods_mk_xcx_card_businesscard_c


def Quartile_anomaly(df, cols):

    print('\n')
    for col in cols:
        df = df.withColumn(col, df[col].cast(DoubleType()))
        quantiles = df.approxQuantile(col, [0.25, 0.75], 0)
        IQR = quantiles[1] - quantiles[0]
        max_value = quantiles[1] + 1.5 * IQR
        print("-" * 20, col, "0.25_quantiles and 0.75_quantiles:",
              quantiles, "Abnormal threshold", max_value)

        df = df.withColumn(col, f.when(f.col(col) > max_value, max_value)
                           .otherwise(f.col(col)))

    return df


def Quartitle_095(df, cols):

    for col in cols:
        df = df.withColumn(col, df[col].cast(DoubleType()))
        quantiles = df.approxQuantile(col, [0.95], 0)
        value_095 = quantiles[0]
        print("-" * 20, col, "0.95_quantiles:", value_095)

        df = df.withColumn(col, f.when(f.col(col) > value_095, value_095)
                           .otherwise(f.col(col)))

    return df


def filter_staff(df_log_action, df_card_businesscard):
    '''
    剔除已创建名片的的用户当作是剔除顾问
    '''
    #
    df_staff_id = df_card_businesscard\
        .select(f.col('cstid')).distinct()

    #
    df_filter_staff = df_log_action\
        .join(df_staff_id, 'cstid', how='left_anti')

    #
    num_cstid = df_log_action\
        .select(f.countDistinct(f.col('cstid')).alias('num'))\
        .collect()[0]['num']
    num_staff = df_staff_id.count()
    num_after_filter_staff = df_filter_staff\
        .select(f.countDistinct(f.col('cstid')).alias('num')) \
        .collect()[0]['num']
    num_click = df_log_action.count()
    num_click_filter_staff = df_filter_staff.count()

    print('the num of cstid is {:,}'.format(num_cstid))
    print('the num of staff is {:,}'.format(num_staff))
    print('the num of cstid filter staff is {:,}'.format(
        num_after_filter_staff))

    print('\nthe num of click is {:,}'.format(num_click))
    print('the num of click filter staff is {:,}'.format(
        num_click_filter_staff))

    return df_filter_staff


def filter_action(df):
    '''
    只筛选部分对意向度有作用的行为
    '''
    #
    num_before_filter = df.count()
    df = df.filter(
        (f.col('code') == 179) |
        (f.col('code') == 305) |
        (f.col('code') == 319) |
        (f.col('code') == 327) |
        (f.col('code') == 342) |
        (f.col('code') == 350) |
        (f.col('code') == 351) |
        (f.col('code') == 364) |
        (f.col('code') == 374) |
        (f.col('code') == 385) |
        (f.col('code').between(392, 419)) |
        (f.col('code') == 439) |
        (f.col('code') == 440)
    )
    num_after_filter = df.count()

    print('\nnum_before_filter is {:,}'.format(num_before_filter))
    print('num_after_filter is {:,}'.format(num_after_filter))

    return df


def weight_action(df):
    '''
    给不同的操作行为赋予不同的意向度权重
    '''

    @pandas_udf('double', PandasUDFType.SCALAR)
    def weight_action_func(pd_series):
        pd_series = pd_series.map(weight.value)
        pd_series.fillna(0.1, inplace=True)

        return pd_series

    weight = {
        179: 0.5,  # 电子物料
        305: 1,  # 项目门户-分享
        319: 2,  # 主题页访问
        327: 4,  # 拼团活动
        342: 1,  # 九宫格分享
        350: 2,  # 助力活动
        351: 5,  # 助力活动-分享
        364: 2,  # 进入页面
        374: 8,  # 提交表单操作
        385: 1.5,  # 文章详情访问
        393: 3,  # 访问楼盘详情页
        394: 1.5,  # 轮播图组件-点击轮播图
        395: 0.5,  # 楼盘简介组件-点击收藏
        396: -1,  # 楼盘简介组件-取消收藏
        397: 1,  # 楼盘详情组件-点击楼盘地址地图
        398: 1,  # 楼盘详情组件-关注变价
        399: 8,  # 楼盘详情组件-关注变价成功
        400: 1,  # 楼盘详情组件-开盘提醒
        401: 8,  # 楼盘详情组件-开盘提醒成功
        402: 2.5,  # 户型组件-点击户型
        403: 2.5,  # 相关资讯组件-点击资讯
        404: 1.5,  # 销售服务组件-点击服务
        405: 2,  # 楼盘周边组件-点击地图
        406: 0.5,  # 销售顾问组件-点击名片
        407: 1,  # 销售顾问组件-拨打电话
        408: 1,  # 底部栏组件-分享好友
        409: 1,  # 底部栏组件-生成海报
        410: 0.5,  # 底部栏组件-销售热线
        411: 1.5,  # 底部栏组件-我要咨询
        412: 0.5,  # 轮播图组件-点击查看更多
        413: 3,  # 物料组件-访问
        414: 0.5,  # 物料组件-点击物料
        415: 2,  # 物料组件-分享
        419: 2.5,  # 户型组件-查看全部
        439: 2,  # 营销工具平台
        440: 2,  # 浏览红包雨
    }
    weight = sc.broadcast(weight)

    df = df \
        .withColumn('weight_action', weight_action_func(f.col('code')))

    return df


def effective_visit(df):
    '''
    统计访客访问楼盘的有效访问次数：
    如访客A在 5月10日10:00 访问了楼盘a, 10:30又访问了楼盘，则10:30那次访问为非有效访问，
    只有间隔在1 hour上，有效访问次数才加1，
    如访客B在 5月10日10:00 访问了楼盘a，中午12:00又访问了改楼盘，则有效访问次数为2
    '''
    window = Window.partitionBy('cstid', 'areaid').orderBy('channelid')
    df = df \
        .withColumn('unix_timestamp', f.unix_timestamp(f.col('channelid'))) \
        .withColumn('shift_unix_timestamp', f.lag(f.col('unix_timestamp')).over(window)) \
        .withColumn('effective_visit', f.when((f.col('unix_timestamp') - f.col('shift_unix_timestamp')) > 3600, 1).otherwise(0)) \
        .withColumn('effective_visit', f.when(f.isnull(f.col('shift_unix_timestamp')), 1).otherwise(f.col('effective_visit'))) \
        .withColumn('effective_visit', f.when(f.isnull(f.col('areaid')), 0).otherwise(f.col('effective_visit'))) \
        .drop('unix_timestamp', 'shift_unix_timestamp')

    return df


def time_on_page(df):
    '''
    折算出访客停留在页面的时间
    '''

    window = Window.partitionBy('cstid').orderBy('channelid')
    df = df \
        .withColumn('unix_timestamp', f.unix_timestamp(f.col('channelid'))) \
        .withColumn('shift_unix_timestamp', f.lag(f.col('unix_timestamp'), -1).over(window)) \
        .withColumn('time_on_page', f.col('shift_unix_timestamp') - f.col('unix_timestamp')) \
        .withColumn('time_on_page', f.when(f.col('time_on_page') > 90, 10).otherwise(f.col('time_on_page'))) \
        .withColumn('time_on_page', f.when(f.isnull(f.col('time_on_page')), 10).otherwise(f.col('time_on_page'))) \
        .drop('unix_timestamp', 'shift_unix_timestamp')

    return df


def feat_eng(df):

    df = weight_action(df)
    df = effective_visit(df)
    df = time_on_page(df)

    return df


def agg_feat(df):
    '''
    生成每一个访客对过去两周有浏览过的楼盘项目的特征，
    输出字段为：
    访客ID  楼盘ID  特征1 特征2 特征3 特征4 。。。

    以下统计范围为过去两周的访问数据：
    Feature：
    - weight_action_sum：访客访问楼盘行为的权重和
    - num_of_visits：访客访问楼盘的有效访问次数
    - total_time：访客访问楼盘的页面总停留时间
    - weight_action_ratio：访客访问该楼盘的行为权重和 / 该访客的所有权重和
    - num_of_visits_ratio：访客访问该楼盘的有效次数 / 该访客的所有访客楼盘和
    - total_time_ratio：访客访问该楼盘的时间 / 该访客的所有访问时间

    '''

    #
    df_train_agg = df\
        .filter(f.col('areaid') != 'NULL')\
        .groupBy("cstid", 'areaid')\
        .agg(
            f.sum(f.col('weight_action')).alias('weight_action_sum'),
            f.sum(f.col("effective_visit")).alias('num_of_visits'),
            f.sum(f.col("time_on_page")).alias("total_time")
        )

    #
    df_train_agg = Quartile_anomaly(
        df_train_agg, cols=['weight_action_sum', 'total_time'])
    df_train_agg = Quartitle_095(df_train_agg, cols=['num_of_visits'])

    #
    win = Window.partitionBy('cstid')
    df_train_agg = df_train_agg\
        .withColumn('weight_action_ratio', f.col('weight_action_sum') / f.sum('weight_action_sum').over(win)) \
        .withColumn('num_of_visits_ratio', f.col('num_of_visits') / f.sum('num_of_visits').over(win)) \
        .withColumn('total_time_ratio', f.col('total_time') / f.sum('total_time').over(win)) \
        .fillna(1, subset=['num_of_visits_ratio', 'total_time_ratio'])

    return df_train_agg


def normalization(df):

    # 向量化
    vector = VectorAssembler(inputCols=[
                             'weight_action_sum', 'num_of_visits', 'total_time'], outputCol='features_vect')
    vector_df = vector.transform(df_train_agg)
    vector_df.cache()

    # 归一化
    mmScaler_df = mmScaler_model.transform(vector_df)
    print('normalization completed!')

    return mmScaler_df


def model(df):

    km = KMeans(
        featuresCol="feature_scaler",
        predictionCol="prediction",
        k=8,
        seed=4,
    )
    km = km.fit(df)
    df = km.transform(df)
    km.write().overwrite().save("./凤凰云/kmeans_model")  # 保存聚类模型，预测时要用
    print('model completed!')

    return df, model


def result_processing(df):

    tmp = df.select('weight_action_sum', 'num_of_visits', 'total_time', 'prediction')\
        .groupby('prediction')\
        .agg(f.mean('weight_action_sum').alias('weight_action_sum'), f.mean('num_of_visits').alias('num_of_visits'), f.mean('total_time').alias('total_time'))\
        .toPandas()  # 转成pandas进行处理
    tmp.set_index('prediction', inplace=True)

    #
    tmp[['weight_action_sum', 'num_of_visits', 'total_time']] = tmp[[
        'weight_action_sum', 'num_of_visits', 'total_time']].rank()
    tmp['sum'] = tmp['weight_action_sum'] + 0.5 * \
        tmp['num_of_visits'] + tmp['total_time']
    tmp['rank'] = tmp['sum'].rank()
    high_intention = list(
        tmp[(tmp['rank'] == 8) | (tmp['rank'] == 7)].index)  # 高意向
    mid_intention = list(
        tmp[(tmp['rank'] == 5) | (tmp['rank'] == 6)].index)  # 中等意向
    small_intention = list(
        tmp[(tmp['rank'] == 3) | (tmp['rank'] == 4)].index)  # 低意向
    not_intention = list(
        tmp[(tmp['rank'] == 1) | (tmp['rank'] == 2)].index)  # 无意向
    d = {}
    d.update(zip(high_intention, [4]*2))   # 高意向
    d.update(zip(mid_intention, [3]*2))    # 中等意向
    d.update(zip(small_intention, [1]*2))  # 低意向
    d.update(zip(not_intention, [0]*2))     # 无意向

    #
    @pandas_udf('double', PandasUDFType.SCALAR)
    def intention_map_func(pd_series):

        pd_series = pd_series.map(d.value)

        return pd_series

    d = sc.broadcast(d)
    df = df \
        .withColumn('intention', intention_map_func(f.col('prediction')))

    return df


if __name__ == '__main__':

    # 
    spark = SparkSession \
            .builder \
            .appName("intention_client_predict") \
            .enableHiveSupport()\
            .getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")  # WARN,ERROR,INFO

    # Enable Arrow-based columnar data transfers
    spark.conf.set("spark.sql.execution.arrow.enabled", "true")

    # 加载数据
    data = load_data()
    ods_mk_xcx_log_action = data[0]
    ods_mk_xcx_arean = data[1]
    ods_mk_xcx_card_businesscard_c = data[2]

    #########################################################

    # 预测时只对最近两周有发生行为记录的访客进行预测，也即一个访客三周前访问过凤凰云，最近两周没有访问，不对其进行意向度预测
    start_date = ods_mk_xcx_log_action\
        .select(f.max(f.col('channelid')).alias('last_channelid'))\
        .select(f.to_date(f.col('last_channelid')).alias('last_date'))\
        .select(f.date_sub(f.col('last_date'), 14).alias('start_date'))\
        .collect()[0]['start_date']
    print('start time:', start_date)

    ods_mk_xcx_log_action = ods_mk_xcx_log_action\
        .filter((f.col('channelid') >= start_date))\
        .coalesce(50)\
        .cache()  # 每次预测时，筛选最近两周的数据进行建模，注意，建模代码是筛选两个月

    df_train = filter_staff(ods_mk_xcx_log_action,
                            ods_mk_xcx_card_businesscard_c).cache()    # 剔除工作人员
    df_train = feat_eng(df_train)  # 生成特征
    df_train = filter_action(df_train)  # 筛选埋点动作
    # 生成训练样本一行为：一个时间窗口一个userid一个areaid 的聚合特征
    df_train_agg = agg_feat(df_train).cache()

    #
    print('\ndf_train_agg_count:{}'.format(df_train_agg.count()))

    #########################################################

    # 读取归一化模型和聚类模型
    mmScaler_model = MinMaxScalerModel.load("./凤凰云/mmScaler_model")
    km = KMeansModel.load("./凤凰云/kmeans_model")
    df_train_agg = normalization(df_train_agg)  # 向量化、归一化
    result = km.transform(df_train_agg)  # 预测
    print('prediction completed!')
    # intention字段等于4代表高意向，3代表中等意向，2代表低意向，0代表无意向
    result = result_processing(result)
    result = result.select('cstid', 'areaid', 'weight_action_sum', 'num_of_visits', 'total_time',
                           'weight_action_ratio', 'num_of_visits_ratio', 'total_time_ratio', 'prediction', 'intention')
    result.write.format("hive").mode("overwrite").saveAsTable(
        'bgy_data_platform.ods_mk_xcx_result')  # 保存到hive表
    result\
        .groupby('intention')\
        .count()\
        .sort('intention')\
        .show()