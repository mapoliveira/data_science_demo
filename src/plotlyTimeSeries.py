#### Function to plot timeseries using plotly ####

import plotly as py

def plotlyTimeSeries(datetimeX, listData, filename):
    import plotly.graph_objs as go
    dataPanda = []
    
    for k, df in enumerate(listData):
        trace = go.Scatter( x = datetimeX,
                y = df.iloc[:,0],
                mode = 'markers+lines',
                name = df.columns[0]
                )
        dataPanda.append(trace)

    # Plot data:
    fig = dict(data=dataPanda)
    
    if filename == '':
        py.offline.plot(fig)
        print('Find interactive graphic in the browser.')
    else:
        print('jupiter notebook output')
        #py.iplot(fig, filename)

       
