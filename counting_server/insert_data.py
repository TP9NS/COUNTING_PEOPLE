import random
from datetime import datetime, timedelta

# 시작 날짜와 종료 날짜 설정
start_date = datetime(2024, 11, 1)
end_date = datetime(2024, 12, 10)

# 지점 설정
locations = ["a_loc", "b_loc", "c_loc"]

# INSERT 문 생성
insert_statements = []
current_date = start_date

while current_date <= end_date:
    for location in locations:
        count = random.randint(100, 500)  # 100~500 사이의 랜덤 값 생성
        date_string = current_date.strftime('%Y-%m-%d %H:%M:%S')  # 날짜를 SQL 포맷으로 변환
        statement = f"INSERT INTO count_p (count, location, date) VALUES ({count}, '{location}', '{date_string}');"
        insert_statements.append(statement)
    current_date += timedelta(days=1)  # 날짜를 하루 증가

# INSERT 문 출력
for stmt in insert_statements:
    print(stmt)
