#### Function to plot timeseries using plotly ####

import plotly

def plotlyTimeSeries(datetimeX, listData, browserOutput=True):
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
    
    if browserOutput == True: 
        plotly.offline.plot(fig)
        print('Find interactive graphic in the browser.')
    else:
        print('jupiter notebook output')
        #plotly.iplot(fig)

       
