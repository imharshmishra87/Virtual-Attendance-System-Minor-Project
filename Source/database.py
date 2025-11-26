import mysql.connector as mc
from datetime import date,datetime
import time
conn=mc.connect(database="test2",password="harsh1514",host="localhost",user="root")
'''Connected with my sql server database'''

class databasedata:
    def __init__(self):
        pass

    def getlecturedetails(self):
        today=date.today()
        dayname=today.strftime('%a')

        query="SELECT start,end FROM timetable WHERE day=%s order by start asc"
        cursor=conn.cursor()

        '''Executing the query'''

        cursor.execute(query,(dayname,))
        return cursor.fetchall()
    
    def givetime(self,start,end,now):
        return start<=now<end
    
    
    def lecturetime(self):
        active=None
        now=datetime.now()
        rows=self.getlecturedetails()

        for row in rows:
            
            start=datetime.combine(date.today(),datetime.min.time())+ row[0]
            end=datetime.combine(date.today(),datetime.min.time())+ row[1]

            if self.givetime(start,end,now):
                return start,end
        return None        
    


            
            
        

