import pymysql
import numpy as np

class DBManager(object):
    def __init__(self):
        self.db = pymysql.connect("localhost", "root", "191969", "faceNet")
        self.cur = self.db.cursor()

    # ***********************************  User table *********************************

    def getAllUserName(self):
        ret = None
        data = []
        try:
            ret = self.cur.execute("select user_name from user_info")
            data = self.cur.fetchall()
        except:
            print("Failed in get User Names")
            return False

        if ret == None or len(data)<1:
            return False
        names = []
        for name in data:
            names.append(name[0])
        return names

    def getUserAvgFeature(self, username):

        if not self.isExistUser(username):
            print("user {} is not exist!".format(username))
            return False

        ret = None
        data = []
        try:
            ret = self.cur.execute("select avg_feature from user_info where user_name=%s", (username,))
            data = self.cur.fetchall()
        except:
            print("Failed to get UserAvgFeature")
            return False

        if(len(data) == 0):
            return []

        if ret == None or len(data)!=1:
            return False

        feature = []
        try:
            feature = np.frombuffer(data[0][0], dtype='float64')
        except:
            print("Failed when buffer to feature")
            return False
        return feature


    def getAllUserAndAvgFeature(self):

        ret = None
        data = []
        try:
            ret = self.cur.execute("select * from user_info")
            data = self.cur.fetchall()
        except:
            print("Failed to get UserAllAvgFeature")
            return False

        if ret == None:
            return False
        if len(data)<1:
            return {}

        #print(np.shape(data))
        #print(len(data))
        info = {}
        try:
            for userdata in data:
                name = userdata[0]
                # print(name)
                # print(userdata[1])
                feature = np.frombuffer(userdata[1], dtype='float64')
                if len(feature) == 0:
                    continue
                info[name]=feature
        except:
            print("Failed when buffer to feature")
            return False
        return info

    ########################## insert user and set avg ##########################

    def insertUser(self, username, avg_feature=None):

        if self.isExistUser(username):
            print("user {} is already exist!".format(username))
            return False

        ret = None
        try:
            if avg_feature == None:
                ret = self.cur.execute("insert into user_info values(%s, %s)", (username, 'null'))
            else:
                bytes = (np.array(avg_feature, dtype='float64')).tostring()
                ret = self.cur.execute("insert into user_info values(%s, %s)", (username, bytes))
            self.db.commit()
        except:
            print("Failed to insert a user")
            return False
        if ret == None:
            return False
        return True

    def setUserAvgFeature(self, username, avg_feature):

        if not self.isExistUser(username):
            print("user {} is not exist!".format(username))
            return False

        ret = None
        try:
            bytes = (np.array(avg_feature, dtype='float64')).tostring()
            ret = self.cur.execute("update user_info set avg_feature=%s where user_name=%s", (bytes, username))
            self.db.commit()
        except:
            print("Failed to set UserAvgFeature")
            return False
        if ret == None:
            return False
        return True

    #**********************************  Feature table ******************************

    def insertFeature(self, username, filename, feature):

        if self.isExistFeature(username, filename):
            print("feature {}-{} is already exist!".format(username, filename))
            return False

        ret = None
        try:

            bytes = (np.array(feature, dtype='float64')).tostring()
            ret = self.cur.execute("insert into feature_data values(null, %s, %s, %s)", (username, filename, bytes))
            self.db.commit()
        except:
            print("Failed to insert a feature")
            return False
        if ret == None:
            return False
        return True

    def getUserFeatures(self, username):

        if not self.isExistUser(username):
            print("user {} is not exist!".format(username))
            return False

        ret = None
        data = []
        try:
            ret = self.cur.execute("select feature from feature_data where user_name=%s", (username,))
            data = self.cur.fetchall()
        except:
            print("Failed to get UserFeature")
            return False

        if ret == None:
            print("ret=None")
            return False

        if len(data)<1:
            print("user [{}] has no features in db".format(username))
            return []

        features = []
        try:
            for bytes in data:
                feature = np.frombuffer(bytes[0], dtype='float64')
                #print(feature)
                features.append(feature)
        except:
            print("Failed when buffer to feature")
            return False

        return features

    def setFeature(self, username, filename, feature):

        if not self.isExistFeature(username, filename):
            print("feature {}-{} is not exist!".format(username, filename))
            return False

        ret = None
        try:
            bytes = (np.array(feature, dtype='float64')).tostring()
            ret = self.cur.execute("update feature_data set feature=%s where "
                                   "user_name=%s and file_name=%s",
                                   (bytes, username, filename))
            self.db.commit()
        except:
            print("Failed to set UserAvgFeature")
            return False
        if ret == None:
            return False
        return True

    def delUserFeatures(self, username):

        if not self.isExistUser(username):
            print("user {} is not exist!".format(username))
            return False

        ret = None
        data = []
        try:
            ret = self.cur.execute("delete from user_info where user_name=%s", (username,))
            self.db.commit()
        except:
            print("Failed in isExist User")
            return False

        if ret == None:
            return False
        return True


    #******************************  Utils ******************************

    def isExistUser(self, username):
        ret = None
        data = []
        try:
            ret = self.cur.execute("select * from user_info where user_name=%s", (username,))
            data = self.cur.fetchall()
        except:
            print("Failed in isExist User")
            return False

        if ret == None or len(data)<1:
            return False
        return True

    def isExistFeature(self, username, filename):
        ret = None
        data = []
        try:
            ret = self.cur.execute("select * from feature_data where "
                                   "user_name=%s and file_name=%s",
                                   (username, filename))
            data = self.cur.fetchall()
        except:
            print("asdasda")
            print("Failed in isExist Feature")
            return False

        if len(data)<1:
            return False
        return True

    def __del__(self):
        # 垃圾回收
        self.db.close()


if __name__ == '__main__':
    a = DBManager()
    print(a.getAllUserName())
    #print(a.delUserFeatures('LarryPage'))
    #dd = a.getUserAvgFeature("adasd")
    # print(dd)
    # ret = a.setUserAvgFeature("adfgfdh",[1,2,3,4])
    # print(a.getUserAvgFeature("sdf"))
    # print(a.isExistUser("wjh"))
    #
    # print(a.insertFeature("asd","0002",[1,2,46,4,5]))
    # print(a.getUserFeatures("asd"))
    # print(a.setFeature('asd','0003',[1,6,8]))
    # print(a.getUserFeatures("asd"))