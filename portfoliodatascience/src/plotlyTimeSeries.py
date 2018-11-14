#### Function to plot timeseries using plotly ####
import plotly
import plotly.plotly as py
import plotly.graph_objs as go

def plotlyTimeSeries(datetimeX, listData, title, divStr=False, jupyter_notebook=False):
    dataPanda = []
    for k, df in enumerate(listData):
        trace = go.Scatter( x = datetimeX,
                y = df.iloc[:,0],
                mode = 'markers+lines',
                name = df.columns[0],
                )
        dataPanda.append(trace)

    # Plot data:
    fig = dict(data=dataPanda)
    fig.update({'layout': {'title': title,'font': dict(size=16)}})
     
    if jupyter_notebook==False:
        print('Find interactive graphic in the browser.')
        plotly.offline.plot(fig)
        
        if divStr:
            div_str = plotly.offline.plot(fig, output_type='div', include_plotlyjs=True)
            return div_str
    else:
        #print('jupiter notebook output')
        from plotly.offline import init_notebook_mode, iplot
        init_notebook_mode(connected=True)
        iplot(fig)
        
        if divStr:
            div_str = plotly.offline.iplot(fig, output_type='div', include_plotlyjs=True)
            return div_str
