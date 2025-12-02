import time
import pandas as pd
from datetime import date,datetime
import mysql.connector as ms
conn=ms.connect(database='test',user='root',host='localhost',password='harsh1514')
cursor=conn.cursor()
today=date.today()
dayname=today.strftime("%a")

class attendance:

    def __init__(self):
        pass

    def getlecturesdata(self):
        query="Select c_id from timetable where day=%s order by start asc"
        values=((dayname),)
        cursor.execute(query,values)
        return cursor.fetchall()
        
    def insertdata(self):
        '''Use this function to store data in the database from your excel sheet that consist of data'''
        df=pd.read_excel(r"D:\projects\Minor Project\Virtual-Attendance-System-Minor-Project\Source\studentdata.xlsx")
        
        for index,rows in df.iterrows():
            query2="Insert into students(students_id,students_names) values (%s,%s)"
            values=(rows['Enrollment No'],rows["Student's Name"])
            cursor.execute(query2,values)
        conn.commit()
        conn.close()
        print("done")

    def takeattendance(self):
        now=datetime.now()
        quer3=""
        # query3="INSERT INTO attendance(name,day) values(%s,{dayname},data[1])"
        
        

obj=attendance()
obj.takeattendance()
