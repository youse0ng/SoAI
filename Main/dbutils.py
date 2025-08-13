import pymysql
# MariaDB 연결 정보
def connect_db():
    '''MariaDB 연결 
        return: conn 
    '''
    conn = pymysql.connect(
        host='localhost',         # DB 호스트 (예: '127.0.0.1')
        user='root',     # DB 사용자 이름
        password='Yousung0528!', # DB 비밀번호
        database='xray',        # 사용할 데이터베이스 이름
        charset='utf8mb4',
        port = 3306,            # 기본 포트
        cursorclass=pymysql.cursors.DictCursor  # 결과를 딕셔너리 형태로 받기
    )
    return conn

# 이름 Index를 이용해서, 환자 정보 조회
def Name_Index(conn, name):
    '''Example 
            {
                'ImagePath': '정상',
                'Class': 5,
                'Caption': 'A plain abdominal radiograph with negative findings.',
                'Age': 6,
                'ImageFile': '../Data/Validation/01.원천데이터/VS_2.정상/5_1901.png',
                'PatientName': '조민성'   
            }    
    '''
    try:
        with conn.cursor() as cursor:
            # 사용자 입력

            # SQL 쿼리 (SQL Injection 방지 방식)
            sql = "SELECT * FROM xray_data WHERE PatientName = %s"
            cursor.execute(sql, (name,))
            results = cursor.fetchall()

            if results:
                return results[0]
            else:
                return f"해당 {name} 이름의 사용자가 없습니다."

    finally:
        conn.close()  