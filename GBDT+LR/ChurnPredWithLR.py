# -*-coding:utf-8-*-
"""
    Author: MagicQ
    Desc:
        LR模型 电信客户流失预测
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import pandas as pd


class ChurnPredWithLR:
    def __init__(self):
        self.file = "../data/telecom-churn/new_churn.csv"
        self.data = self.load_data
        self.train, self.test = self.split

    # 加载数据
    @property
    def load_data(self):
        # 同时做one-hot编码，需要做one-hot编码的列（从0列开始） [1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17]
        # gender,SeniorCitizen,Partner,Dependents,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod
        data = pd.read_csv(self.file)
        # print(type(data))
        labels = list(data.keys())
        # print(len(labels))
        # 构建labels 和对应的value映射
        fDict = dict()
        for f in labels:
            if f not in ['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']:
                fDict[f] = sorted(list(data.get(f).unique()))
        # 写入文件
        fw = open("../data/telecom-churn/one_hot_churn.csv", "w")
        fw.write("customerID,")
        for i in range(1, 47): fw.write('f_%s,' % i)
        fw.write("Churn\n")
        for line in data.values:
            list_line = list(line)
            # 存放一行 one hot编码后的结果
            list_result = list()
            for i in range(0, list_line.__len__()):
                if labels[i] in ['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']:
                    list_result.append(list_line[i])
                else:
                    # 创建one hot数组，看该labei下对应多少个不同的值
                    arr = [0] * fDict[labels[i]].__len__()
                    # 值的下标
                    ind = fDict[labels[i]].index(list_line[i])
                    # 让对应位置为1，其余位置为0
                    arr[ind] = 1
                    for one in arr:  list_result.append(one)
            fw.write(",".join([str(f) for f in list_result]) + "\n")
        fw.close()
        return pd.read_csv("../data/telecom-churn/one_hot_churn.csv")

    # 拆分数据集
    @property
    def split(self):
        train, test = train_test_split(
            self.data,
            test_size=0.1,
            random_state=40
        )
        return train, test

    # 模型训练
    def train_model(self):
        print("Start Train Model ... ")
        lable = "Churn"
        ID = "customerID"
        x_columns = [x for x in self.train.columns if x not in [lable, ID]]
        x_train = self.train[x_columns]
        y_train = self.train[lable]
        # 定义模型
        lr = LogisticRegression(penalty="l2", tol=1e-4, fit_intercept=True)
        lr.fit(x_train, y_train)
        return lr

    # 模型评估
    def evaluate(self, lr, type):
        lable = "Churn"
        ID = "customerID"
        x_columns = [x for x in self.test.columns if x not in [lable, ID]]
        x_test = self.test[x_columns]
        y_test = self.test[lable]
        if type == 1:
            y_pred = lr.predict(x_test)
            new_y_pred = y_pred
        elif type == 2:
            y_pred = lr.predict_proba(x_test)
            new_y_pred = list()
            for y in y_pred:
                new_y_pred.append(1 if y[1] > 0.5 else 0)
        mse = mean_squared_error(y_test, new_y_pred)
        print("MSE: %.4f" % mse)
        accuracy = metrics.accuracy_score(y_test.values, new_y_pred)
        print("Accuracy : %.4g" % accuracy)
        auc = metrics.roc_auc_score(y_test.values, new_y_pred)
        print("AUC Score : %.4g" % auc)


if __name__ == "__main__":
    pred = ChurnPredWithLR()
    lr = pred.train_model()
    # type=1：表示输出0 、1 type=2：表示输出概率
    pred.evaluate(lr, type=2)
