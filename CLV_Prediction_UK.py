import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5*interquantile_range
    low_limit = quartile1 - 1.5*interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.shape
df.head()
df.info()
df.describe().T

df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")
df.describe().T

df["TotalPrice"] = df["Quantity"] * df["Price"]
df.head()
df["InvoiceDate"].max()
today_date = df["InvoiceDate"].max() + dt.timedelta(days=1)

#############################################
# RFM Table
#############################################

rfm = df.groupby("Customer ID").agg({"InvoiceDate": [lambda date : (date.max()-date.min()).days,
                                                     lambda date : (today_date - date.min()).days],
                                     "Invoice": lambda num : num.nunique(),
                                     "TotalPrice": lambda TotalPrice : TotalPrice.sum()
                                     })
rfm.columns = rfm.columns.droplevel(0)

## recency_cltv_p
rfm.columns = ['recency_cltv_p', 'T', 'frequency', 'monetary']

## basitleştirilmiş monetary_avg
rfm["monetary"] = rfm["monetary"] / rfm["frequency"]

rfm.rename(columns = {"monetary": "monetary_avg"}, inplace = True)

# BGNBD için WEEKLY RECENCY VE WEEKLY T'nin HESAPLANMASI
## recency_weekly_p
rfm["recency_weekly_p"] = rfm["recency_cltv_p"] / 7
rfm["T_weekly"] = rfm["T"] / 7

# KONTROL
rfm = rfm[rfm["monetary_avg"] > 0]

## freq > 1
rfm = rfm[(rfm['frequency'] > 1)]
rfm["frequency"] = rfm["frequency"].astype(int)

##############################################################
# 2. BG/NBD Modelinin Kurulması
##############################################################
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(rfm["frequency"],
        rfm["recency_weekly_p"],
        rfm["T_weekly"])

################################################################
# 1 hafta içinde en çok satın alma beklediğimiz 10 müşteri kimdir?
################################################################
bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        rfm["frequency"],
                                                        rfm["recency_weekly_p"],
                                                        rfm["T_weekly"]).sort_values(ascending=False).head(10)

rfm["expected_number_of_purchases"] = bgf.predict(1,
                                                  rfm["frequency"],
                                                  rfm["recency_weekly_p"],
                                                  rfm["T_weekly"])
rfm.head(10)

################################################################
# 1 ay içinde en çok satın alma beklediğimiz 10 müşteri kimdir?
################################################################

bgf.predict(4,
            rfm['frequency'],
            rfm['recency_weekly_p'],
            rfm['T_weekly']).sort_values(ascending=False).head(10)

rfm["expected_number_of_purchases"] = bgf.predict(4,
                                                  rfm['frequency'],
                                                  rfm['recency_weekly_p'],
                                                  rfm['T_weekly'])

rfm.sort_values("expected_number_of_purchases", ascending=False).head(20)

################################################################
# 1 Ay içinde tüm Şirketin Beklenen Satış Sayısı Nedir?
################################################################

bgf.predict(4,
            rfm['frequency'],
            rfm['recency_weekly_p'],
            rfm['T_weekly']).sum()

################################################################
# 3 Ayda Tüm Şirketin Beklenen Satış Sayısı Nedir?
################################################################

bgf.predict(12,
            rfm['frequency'],
            rfm['recency_weekly_p'],
            rfm['T_weekly']).sum()

################################################################
# Tahmin Sonuçlarının Değerlendirilmesi
################################################################

plot_period_transactions(bgf)
plt.show()

##############################################################
# 3. GAMMA-GAMMA Modelinin Kurulması
##############################################################

ggf = GammaGammaFitter(penalizer_coef= 0.01)
ggf.fit(rfm["frequency"], rfm["monetary_avg"])

ggf.conditional_expected_average_profit(rfm["frequency"],
                                        rfm["monetary_avg"]).sort_values(ascending=False).head(10)

rfm["expected_average_profit"] = ggf.conditional_expected_average_profit(rfm['frequency'],
                                                                         rfm['monetary_avg'])
rfm.sort_values("expected_average_profit", ascending=False).head(20)

##############################################################
# 4. BG-NBD ve GG modeli ile CLTV'nin hesaplanması.
##############################################################

cltv = ggf.customer_lifetime_value(bgf,
                                   rfm["frequency"],
                                   rfm["recency_weekly_p"],
                                   rfm["T_weekly"],
                                   rfm["monetary_avg"],
                                   time=6, #6aylık
                                   freq= "W", #T nin freq bilgisi
                                   discount_rate = 0.01)
cltv.head()
cltv = cltv.reset_index()
cltv.sort_values(by="clv", ascending=False).head(25)
rfm_cltv_final = rfm.merge(cltv, on="Customer ID", how="left")
rfm_cltv_final.head(15)

cltv1 = ggf.customer_lifetime_value(bgf,
                                    rfm['frequency'],
                                    rfm['recency_weekly_p'],
                                    rfm['T_weekly'],
                                    rfm['monetary_avg'],
                                    time=1,  # 1 aylık
                                    freq="W",  # T'nin frekans bilgisi.
                                    discount_rate=0.01)

cltv1 = cltv1.reset_index()
cltv1.sort_values(by="clv", ascending=False).head()
rfm_cltv1_final = rfm.merge(cltv1, on="Customer ID", how="left")
rfm_cltv1_final.head()

rfm_cltv1_final.sort_values(by="clv", ascending=False).head()

# 12 aylık

cltv12 = ggf.customer_lifetime_value(bgf,
                                     rfm['frequency'],
                                     rfm['recency_weekly_p'],
                                     rfm['T_weekly'],
                                     rfm['monetary_avg'],
                                     time=12,  # 12 aylık
                                     freq="W",  # T'nin frekans bilgisi.
                                     discount_rate=0.01)

cltv12 = cltv12.reset_index()
cltv1.sort_values(by="clv", ascending=False).head()
rfm_cltv12_final = rfm.merge(cltv12, on="Customer ID", how="left")
rfm_cltv12_final.head()

df.loc[df["Customer ID"] == 17850, "Invoice"].nunique()
# 1. 2010-2011 UK müşterileri için 6 aylık CLTV'ye göre' \
#                                              ' tüm müşterilerinizi 3 gruba (segmente) ayırınız ' \
#                                              've grup isimlerini veri setine ekleyiniz. Örneğin (A, B, C)
rfm_cltv_final["Segment"] = pd.qcut(rfm_cltv_final["clv"], 3, labels=["C", "B", "A"])
rfm_cltv_final.head(50)
rfm_cltv_final["Segment"].value_counts()

rfm_cltv_final.groupby("Segment")[["frequency", "monetary_avg", "recency_weekly_p", "clv", "T_weekly"]].\
                agg({"count", "mean", "sum"})

rfm_cltv_final.columns
# 2. 2. top_flag adında bir değişken oluşturunuz. CLTV'ye göre en iyi yüzde 20'yi seçiniz ve bu kişiler için top_flag 1
# yazınız. Diğerlerine 0 yazınız.
rfm_cltv_final["top_flag"] = pd.qcut(rfm_cltv_final["clv"], 5, labels=["E","D","C", "B", "A"])
rfm_cltv_final.head()
rfm_cltv_final["top_flag"] = ["top_flag_1" if value == "A" else 0 for value in rfm_cltv_final["top_flag"].values]
rfm_cltv_final[rfm_cltv_final["top_flag"]=="top_flag_1"].head(20)

## Bu 3 segment için weekly receny değeri yakın değerler almıştır
#frekans, ortalama monetary değeri birbirinden farklıdır.En son saydığımız değerler
#segmentlere göre hayli farklılık göstermektedir. Segmentlerdeki müşterilerin
#haftalık alışveriş yakınlığı segment ayırmak için yeterli bir ölçüt olammakla beraber
#diğer ölçütler gayet yeterlidir
