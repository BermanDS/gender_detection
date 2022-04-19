from model_predict import *
from flask import (
                    Flask,
                    Response,
                    render_template,
                    session,
                    flash,
                    request,
                    json,
                    )
import os

app = Flask(__name__,
            static_url_path='', 
            static_folder='static',
            template_folder='templates'
            )


@app.route('/', methods=['GET', 'POST'])
def index():
    """
    """
    
    kind = 'gender_detection'
    location = os.getcwd()
    
    if request.method == 'POST':
        try:
            date_start = parse(request.form['start'])
            date_end = parse(request.form['end'])
        except:
            date_start = (datetime.today() - timedelta(14))
            date_end = datetime.today()
    else:
        date_start = (datetime.today() - timedelta(14))
        date_end = datetime.today()
    
    if date_start > date_end:
        date_end = date_start + timedelta(14)
    
    pattern= re.escape(template_logs)
    pattern= re.sub(r'\\\$(\w+)', r'(?P<\1>.*)', pattern)
    date_ranges = ["1970-01-01", datetime.today().strftime("%Y-%m-%d")]
    ls, dats, times, source, types, kinds, mesagges, colors = [], [], [], [], [], [], [], []
    df_log = pd.DataFrame()
    
    for path_file in [x for x in os.listdir(location) if '.log' in x]:
        if os.path.exists(os.path.join(location, path_file)):
                
            with open(os.path.join(location, path_file)) as f:
                log_ls = [x  for x in f.readlines() if re.match(pattern, x)]
            df_log_tmp = pd.DataFrame([re.match(pattern, string).groupdict() for string in log_ls if re.match(pattern, string)])

            if df_log_tmp.empty: continue

            if df_log.empty:
                df_log = df_log_tmp.copy()
            else:
                df_log = df_log.merge(df_log_tmp, how = 'outer')
    
    if not df_log.empty:
        df_log['date'] = pd.to_datetime(df_log['date'], errors='coerce')
        df_log.sort_values(by = 'date', inplace = True)
        date_ranges[0] = df_log.loc[~df_log['date'].isnull(), 'date'].min().strftime("%Y-%m-%d")
        df_log = df_log.loc[(~df_log['date'].isnull())&\
                            (df_log['date'] >= date_start)&\
                            (df_log['date'] <= date_end)] 
        df_log['time'] = df_log['date'].apply(lambda x: x.strftime('%H:%M:%S'))
        df_log['date'] = df_log['date'].apply(lambda x: x.strftime('%Y/%m/%d'))
        df_log['color'] = df_log['type'].map({"INFO":'darkgreen','WARNING':'darkgoldenrod','ERROR':'darkred'})
        ls = list(range(len(df_log)))
        dats = df_log['date'].values
        times = df_log['time'].values
        source = df_log['source'].values
        types = df_log['type'].values
        kinds = df_log['kind'].values
        mesagges = df_log['msg'].values
        colors = df_log['color'].values

    return render_template('index.html', title = f'APP - {kind}', ls = ls, dats = dats, source = source, types = types,
                           kinds = kinds, mesagges = mesagges, color = colors, times = times, date_select = date_ranges)


@app.route('/<kind>/v<version>/predict_data', methods=['POST'])
def gender_predict(kind, version):
    """
    Procedure for uploading input data to kafka
    """

    logg = get_init_logging('rest_processing')

    if kind == os.environ.get('ACCESS_NAME'):
        ### constraint of names count
        try:
            constraint_volume = int(os.environ.get('COUNT'))
        except Exception as err:
            logg.warning(f"Problem with parsing constraint variable : {err}. Assigned default value = 100", \
                        kind = 'request proccessing')
            constraint_volume = 100
    else:
        return Response(response=json.dumps({"Message": f"Incorrect request uri: {kind}, please correct url."}),\
                        status=404,\
                        mimetype='application/json')

    
    try:
        input_data = request.json
        if not isinstance(input_data, dict):
            input_data = None
    except Exception as err:
        logg.error(f"Problem with parsing input json : {err}", kind = 'request proccessing')    
        input_data = None

    ###_checking content availability
    if input_data is None or input_data == {} or 'data' not in input_data:
        logg.error("Empty request", kind = 'request proccessing')
        return Response(response=json.dumps({"Message": "Please provide data field"}),
                        status=400,
                        mimetype='application/json')
    
    ###_checking format of input data
    elif not isinstance(input_data['data'], list):
        logg.error("Data mismatch in request", kind = 'request proccessing')
        return Response(response=json.dumps({"Message": "Mismatch data type of input data, please wrap data to list of values"}),
                        status=415,
                        mimetype='application/json')
    
    ###_checking constraints of volume data
    elif len(input_data['data']) > constraint_volume:
        logg.warning("Too long volume request", kind = 'request proccessing')
        return Response(response=json.dumps({"Message": f"Constraint on length of input data {constraint_volume}, please decrease volume of data"}),
                        status=507,
                        mimetype='application/json')

    elif isinstance(input_data['data'], list):
        
        ###_checking language
        if hascyrillic(''.join(map(str, input_data['data']))):
            language = 'rus'
        else:
            language = 'eng'
        
        try:
            result = get_prediction(language, input_data['data'])
            logg.info("Made response", kind = 'request proccessing')
            
            return Response(response=json.dumps(result, cls = NpEncoder),
                            status=200,
                            mimetype='application/json')
        except Exception as err:
            logg.error(f"Problem with prediction : {err}", kind = 'request proccessing')
            return Response(response=json.dumps({"Message": "Couldn't accept data"}),
                            status=406,
                            mimetype='application/json')
    
    return Response(response=json.dumps({"Message": "Couldn't accept data"}),
                    status=406,
                    mimetype='application/json')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=os.environ.get('ACCESS_PORT'), debug = False)
    