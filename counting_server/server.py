from flask import Flask, request, render_template, redirect, url_for, send_file,jsonify
import mysql.connector
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io

app = Flask(__name__)
location_counts={}
# MySQL 데이터베이스 연결 설정
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'psh0811',
    'database': 'counting'
}

# MySQL 데이터베이스 연결 함수
def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host=db_config['host'],
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['database']
        )
        return conn
    except mysql.connector.Error as err:
        print(f"DB 연결 실패: {err}")
        raise

@app.route('/', methods=['GET', 'POST'])
def main():
    conn = None
    cursor = None
    results = []
    recent_counts = []
    search_by_location_results = []
    days_filter = 30  # 기본 값: 최근 한 달 데이터
    start_date = None
    end_date = None

    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # 버튼 클릭에 따른 날짜 필터
        if request.method == 'POST':
            if 'last_3_days' in request.form:
                days_filter = 3
            elif 'last_week' in request.form:
                days_filter = 7
            elif 'last_month' in request.form:
                days_filter = 30

            # 날짜 범위 필터 검색
            elif 'search_by_date' in request.form:
                start_date = request.form.get('start_date')
                end_date = request.form.get('end_date')

                if not start_date or not end_date:
                    return "시작 날짜와 종료 날짜를 모두 입력해야 합니다.", 400

                query = """
                    SELECT location, SUM(count) AS total_count 
                    FROM count_p 
                    WHERE date BETWEEN %s AND %s 
                    GROUP BY location
                """
                cursor.execute(query, (start_date, end_date))
                results = cursor.fetchall()

            # 지점 이름으로 검색
            elif 'search_by_location' in request.form:
                location = request.form.get('location')

                if not location:
                    return "지점 이름을 입력해야 합니다.", 400

                query = """
                    SELECT location, count, date 
                    FROM count_p 
                    WHERE location = %s 
                    ORDER BY date DESC
                """
                cursor.execute(query, (location,))
                search_by_location_results = cursor.fetchall()

        # 버튼 클릭 시 기본 필터 적용
        if not results and not search_by_location_results:
            start_date = (datetime.now() - timedelta(days=days_filter)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
            query = """
                SELECT location, SUM(count) AS total_count 
                FROM count_p 
                WHERE date >= %s 
                GROUP BY location
            """
            cursor.execute(query, (start_date,))
            recent_counts = cursor.fetchall()

    except Exception as e:
        print(f"Error querying data: {e}")
        return f"Error occurred: {e}", 500

    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()

    return render_template(
        'main.html',
        location_counts=recent_counts,
        results=results,
        search_by_location_results=search_by_location_results,
        days_filter=days_filter,
        start_date=start_date,
        end_date=end_date
    )


# 분석 페이지
@app.route('/analyze/<location>')
def analyze(location):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # 지점 이름으로 데이터 조회
        query = """
            SELECT date, count 
            FROM count_p 
            WHERE location = %s 
            ORDER BY date ASC
        """
        cursor.execute(query, (location,))
        data = cursor.fetchall()

        if not data:
            return f"{location}에 대한 데이터가 없습니다.", 404

        # 날짜와 출입자 수 분리
        dates = [entry['date'] for entry in data]
        counts = [entry['count'] for entry in data]

        # Matplotlib 백엔드 설정
        import matplotlib
        matplotlib.use('Agg')

        # 그래프 생성 (Bar)
        plt.figure(figsize=(10, 6))
        x_positions = range(len(dates))  # X축 위치 설정
        plt.bar(x_positions, counts, color='skyblue', label='NUMBER OF PEOPLE')

        # X축 레이블 수동 설정 (날짜 포맷 변경)
        formatted_dates = [date.strftime('%Y-%m-%d') for date in dates]
        plt.xticks(x_positions, formatted_dates, rotation=45)

        plt.title(f"{location} ANALATICS")
        plt.xlabel("DATE")
        plt.ylabel("COUNT")
        plt.legend()
        plt.tight_layout()

        # 그래프를 메모리에 저장
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return send_file(buf, mimetype='image/png')

    except Exception as e:
        print(f"Error during analysis: {e}")
        return f"Error occurred: {e}", 500

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if request.method == 'POST':
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            # 지점별 데이터를 데이터베이스에 저장
            for location, count in location_counts.items():
                cursor.execute(
                    "INSERT INTO count_p (count, location, date) VALUES (%s, %s, %s)",
                    (count, location, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                )
            conn.commit()
            print("모든 데이터 저장 성공")
            
            # 저장 후 데이터를 초기화
            location_counts.clear()

        except Exception as e:
            print(f"Error saving data: {e}")
            return f"Error occurred: {e}", 500

        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    return render_template('dashboard.html', location_counts=location_counts)
@app.route('/person', methods=['POST'])
def receive_person_data():
    try:
        data = request.get_json()
        area_name = data.get("areaName")
        if not area_name:
            return jsonify({"error": "areaName is required"}), 400

        # 해당 지점의 카운트 업데이트
        global location_counts
        location_counts[area_name] = location_counts.get(area_name, 0) + 1
        print(f"Updated count for {area_name}: {location_counts[area_name]}")
        return jsonify({"message": "Data received", "current_count": location_counts[area_name]}), 200
    except Exception as e:
        print(f"Error receiving person data: {e}")
        return jsonify({"error": str(e)}), 500
@app.route('/analyze_multiple', methods=['GET'])
def analyze_multiple():
    conn = None
    cursor = None
    try:
        # 시작 날짜와 종료 날짜를 쿼리 매개변수에서 가져오기
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        if not start_date or not end_date:
            return "시작 날짜와 종료 날짜를 지정해야 합니다.", 400

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # 지정된 날짜 범위에서 모든 지점의 데이터를 가져오기
        query = """
            SELECT location, SUM(count) AS total_count 
            FROM count_p 
            WHERE date BETWEEN %s AND %s 
            GROUP BY location
            ORDER BY location
        """
        cursor.execute(query, (start_date, end_date))
        data = cursor.fetchall()

        if not data:
            return "해당 기간 동안 데이터가 없습니다.", 404

        # 지점 이름과 출입자 수 분리
        locations = [entry['location'] for entry in data]
        counts = [entry['total_count'] for entry in data]

        # Matplotlib 백엔드 설정
        import matplotlib
        matplotlib.use('Agg')

        # 그래프 생성 (Bar Chart)
        plt.figure(figsize=(10, 6))
        plt.bar(locations, counts, color='skyblue')
        plt.title(f"{start_date} ~ {end_date} analatics")
        plt.xlabel("location")
        plt.ylabel("count")
        plt.xticks(rotation=45)
        plt.tight_layout()

        # 그래프를 메모리에 저장
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return send_file(buf, mimetype='image/png')

    except Exception as e:
        print(f"Error during analysis: {e}")
        return f"Error occurred: {e}", 500

    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)