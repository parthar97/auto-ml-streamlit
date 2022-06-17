import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from pyarrow import csv
import numpy as np

st.write("## Seaborn App")
with st.expander('Tutorial'):
    st.video('https://www.youtube.com/watch?v=ElMLZ7BoSpg')
uploaded_file = st.sidebar.file_uploader("Upload your input CSV/XLSX file", type=["csv","xlsx"])
# if uploaded_file is not None:
@st.experimental_memo
def read_file(uploaded_file):
        try:
            dat=csv.read_csv(uploaded_file)
            data2=dat.to_pandas()  
            data2.dropna()
        except:
            data2=pd.read_excel(uploaded_file)
        quant= [col for col in data2.columns if data2.dtypes[col]!='object']
        qual= [col for col in data2.columns if data2.dtypes[col]=='object']
        quali=list()
        for i in range(len(qual)):
            if len(set(data2[qual[i]]))<=50:
                quali.append(qual[i])
        qual=quali
        quant1=[0*1 for i in range(len(quant)+1)]
        for i in range(len(quant1)):
                if i==0:
                    quant1[i]=None
                else:
                    quant1[i]=quant[i-1]
        qual1=[0*1 for i in range(len(qual)+1)]
        for i in range(len(qual1)):
                if i==0:
                    qual1[i]=None
                else:
                    qual1[i]=qual[i-1]
        return data2,quant,qual,quant1,qual1
try:
    data2,quant,qual,quant1,qual1=read_file(uploaded_file)
    plot_type=[None,'Relational Plot','Distribution Plot','Categorical Plot','Pair Plot','Joint Plot','Heat Map','Correlation Map']
    ctype=st.selectbox('Chart Type',plot_type)
    if ctype=='Relational Plot':   
        try:
            x= st.selectbox('Select X',quant1)
            y= st.selectbox('Select Y',quant1)            
            kind=st.sidebar.selectbox('Kind',['scatter','line'])
            row= st.sidebar.selectbox('Select Row',qual1)
            col= st.sidebar.selectbox('Select Col',qual1)
            height= st.sidebar.slider('Height',1.0,20.0,8.0)
            aspect= st.sidebar.slider('Aspect',0.5,5.0,1.5)
            alpha= st.sidebar.slider('Alpha',0.0,1.0,1.0)
            hue=st.selectbox('Hue',qual1)
            size=st.sidebar.selectbox('Size',qual1)
            style=st.sidebar.selectbox('Style',qual1)
            palette=st.sidebar.selectbox('Palette',['rocket','mako','crest','flare','magma','viridis','rocket_r','cubehelix','Blues','icefire','Spectral','coolwarm'])
            context=st.sidebar.selectbox('Context',['paper', 'notebook', 'talk', 'poster'])
            set_style=st.sidebar.selectbox('Style',['white','whitegrid','dark','darkgrid'])

            r1=st.radio('Generate Plot',['n','y'])
            if r1=='y':
                sns.set_context("{}".format(context), rc={"font.size":9,"axes.titlesize":12,"axes.labelsize":14}) 
                sns.set_style("{}".format(set_style))
                s=sns.relplot(data=data2,kind=kind, x=x, y=y,hue=hue,row=row,col=col,style=style,size=size,legend='auto',height=height,aspect=aspect,alpha=alpha,palette=palette)
                try:
                    s.fig.suptitle('{} VS {}'.format(''.join(x),''.join(y)).upper(),fontsize=16,y=1.02, fontweight='bold')
                    with st.container():
                        fig=plt.savefig('seaborn.png',bbox_inches='tight',dpi=200)
                        st.pyplot(fig=plt)   
                        with open("seaborn.png", "rb") as file:
                             btn = st.download_button(label="Download image",data=file,file_name="seaborn.png",mime="image/png")
                except:
                    with st.container():
                        fig=plt.savefig('seaborn.png',bbox_inches='tight',dpi=200)
                        st.pyplot(fig=plt)   
                        with open("seaborn.png", "rb") as file:
                             btn = st.download_button(label="Download image",data=file,file_name="seaborn.png",mime="image/png")
        except:
                st.warning('Choose Something')
    if ctype=='Distribution Plot':   
        try:
            x= st.selectbox('Select X',quant1)
            y= st.selectbox('Select Y',quant1)
            kind=st.sidebar.selectbox('Kind',['hist','kde','ecdf'])
            row= st.sidebar.selectbox('Select Row',qual1)
            col= st.sidebar.selectbox('Select Col',qual1)
            height= st.sidebar.slider('Height',1.0,20.0,8.0)
            aspect= st.sidebar.slider('Aspect',0.5,5.0,1.5)
            alpha= st.sidebar.slider('Alpha',0.0,1.0,1.0)
            hue=st.selectbox('Hue',qual1)
            palette=st.sidebar.selectbox('Palette',['rocket','mako','crest','flare','magma','viridis','rocket_r','cubehelix','Blues','icefire','Spectral','coolwarm'])
            context=st.sidebar.selectbox('Context',['paper', 'notebook', 'talk', 'poster'])
            set_style=st.sidebar.selectbox('Style',['white','whitegrid','dark','darkgrid'])

            r2=st.radio('Generate Plot',['n','y'])
            if r2=='y':
                sns.set_context("{}".format(context), rc={"font.size":9,"axes.titlesize":12,"axes.labelsize":14}) 
                sns.set_style("{}".format(set_style))
                s=sns.displot(data=data2,kind=kind,x=x, y=y, row=row, col=col, hue=hue,
                              palette=palette,alpha=alpha,weights=None,rug=True,rug_kws=None,
                              log_scale=None,legend='auto',hue_order=None,hue_norm=None,color=None,
                              col_wrap=None,row_order=None,col_order=None,height=height, aspect=aspect, 
                              facet_kws=None) 
                try:
                    s.fig.suptitle('{} VS {}'.format(''.join(x),''.join(y)).upper(),fontsize=16,y=1.02, fontweight='bold')
                    with st.container():
                        fig=plt.savefig('seaborn.png',bbox_inches='tight',dpi=200)
                        st.pyplot(fig=plt)   
                        with open("seaborn.png", "rb") as file:
                             btn = st.download_button(label="Download image",data=file,file_name="seaborn.png",mime="image/png")
                except:
                    with st.container():
                        fig=plt.savefig('seaborn.png',bbox_inches='tight',dpi=200)
                        st.pyplot(fig=plt)   
                        with open("seaborn.png", "rb") as file:
                             btn = st.download_button(label="Download image",data=file,file_name="seaborn.png",mime="image/png")
        except:
                st.warning('Choose Something')
    if ctype=='Categorical Plot':   
        try:
            x= st.selectbox('Select X',qual1)
            y= st.selectbox('Select Y',quant1)
            kind=st.sidebar.selectbox('Kind',['strip','swarm','box','violin','boxen','point','count'])
            row= st.sidebar.selectbox('Select Row',qual1)
            col= st.sidebar.selectbox('Select Col',qual1)
            height= st.sidebar.slider('Height',1.0,20.0,8.0)
            aspect= st.sidebar.slider('Aspect',0.5,5.0,1.5)
            hue=st.selectbox('Hue',qual1)
            palette=st.sidebar.selectbox('Palette',['rocket','mako','crest','flare','magma','viridis','rocket_r','cubehelix','Blues','icefire','Spectral','coolwarm'])
            context=st.sidebar.selectbox('Context',['paper', 'notebook', 'talk', 'poster'])
            set_style=st.sidebar.selectbox('Style',['white','whitegrid','dark','darkgrid'])

            r3=st.radio('Generate Plot',['n','y'])
            if r3=='y':
                sns.set_context("{}".format(context), rc={"font.size":9,"axes.titlesize":12,"axes.labelsize":14}) 
                sns.set_style("{}".format(set_style))
                s=sns.catplot(data=data2,kind=kind,x=x, y=y, row=row, col=col, hue=hue,palette=palette,height=height,aspect=aspect,legend='auto',orient='v') 
                try:
                    s.fig.suptitle('{} VS {}'.format(''.join(x),''.join(y)).upper(),fontsize=16,y=1.02,fontweight='bold')
                    with st.container():
                        fig=plt.savefig('seaborn.png',bbox_inches='tight',dpi=200)
                        st.pyplot(fig=plt)   
                        with open("seaborn.png", "rb") as file:
                             btn = st.download_button(label="Download image",data=file,file_name="seaborn.png",mime="image/png")
                except:
                    with st.container():
                        fig=plt.savefig('seaborn.png',bbox_inches='tight',dpi=200)
                        st.pyplot(fig=plt)   
                        with open("seaborn.png", "rb") as file:
                             btn = st.download_button(label="Download image",data=file,file_name="seaborn.png",mime="image/png")
        except:
               st.warning('Choose Something')
    if ctype=='Pair Plot':   
        try:
            x_vars1= st.multiselect('Select X',quant)
            y_vars1= st.multiselect('Select Y',quant)
            kind=st.sidebar.selectbox('Kind',['scatter', 'kde', 'hist', 'reg'])
            diag_kind=st.sidebar.selectbox('Diagnol Kind',['auto','hist','kde'])
            corner= st.sidebar.selectbox('Select Corner',['n','y'])
            height= st.sidebar.slider('Height',1.0,20.0,8.0)
            aspect= st.sidebar.slider('Aspect',0.5,5.0,1.5)
            hue=st.selectbox('Hue',qual1)
            palette=st.sidebar.selectbox('Palette',['rocket','mako','crest','flare','magma','viridis','rocket_r','cubehelix','Blues','icefire','Spectral','coolwarm'])
            context=st.sidebar.selectbox('Context',['paper', 'notebook', 'talk', 'poster'])
            set_style=st.sidebar.selectbox('Style',['white','whitegrid','dark','darkgrid'])

            r4=st.radio('Generate Plot',['n','y'])
            if r4=='y':
                if corner=='n':
                    sns.set_context("{}".format(context), rc={"font.size":9,"axes.titlesize":12,"axes.labelsize":14}) 
                    sns.set_style("{}".format(set_style))
                    s=sns.pairplot(data=data2,kind=kind,diag_kind=diag_kind,x_vars=x_vars1,y_vars=y_vars1,hue=hue,height=height,aspect=aspect)
                elif corner=='y' and len(x_vars1)==len(y_vars1):
                    sns.set_context("{}".format(context), rc={"font.size":9,"axes.titlesize":12,"axes.labelsize":14}) 
                    sns.set_style("{}".format(set_style))
                    s=sns.pairplot(data=data2,kind=kind,diag_kind=diag_kind,x_vars=x_vars1,y_vars=y_vars1,hue=hue,height=height,aspect=aspect,corner=True)
                elif corner=='y' and len(x_vars1)!=len(y_vars1):
                    st.warning('Corner not available for non-square plots')

                try:
                    s.fig.suptitle("PAIR PLOT",x=0.43,y=1.01,ha='left',va='baseline',fontsize=16,fontweight='bold')
                    with st.container():
                        fig=plt.savefig('seaborn.png',bbox_inches='tight',dpi=200)
                        st.pyplot(fig=plt)   
                        with open("seaborn.png", "rb") as file:
                             btn = st.download_button(label="Download image",data=file,file_name="seaborn.png",mime="image/png")
                except:
                    with st.container():
                        fig=plt.savefig('seaborn.png',bbox_inches='tight',dpi=200)
                        st.pyplot(fig=plt)   
                        with open("seaborn.png", "rb") as file:
                             btn = st.download_button(label="Download image",data=file,file_name="seaborn.png",mime="image/png")
        except:
               st.warning('Choose Something')
    if ctype=='Joint Plot':   
        try:
            x= st.selectbox('Select X',quant)
            y= st.selectbox('Select Y',quant1)
            kind=st.sidebar.selectbox('Kind',['scatter', 'kde', 'hist', 'hex', 'reg', 'resid'])
            height= st.sidebar.slider('Height',1.0,20.0,8.0)
            hue=st.selectbox('Hue',qual1)
            palette=st.sidebar.selectbox('Palette',['rocket','mako','crest','flare','magma','viridis','rocket_r','cubehelix','Blues','icefire','Spectral','coolwarm'])
            context=st.sidebar.selectbox('Context',['paper', 'notebook', 'talk', 'poster'])
            set_style=st.sidebar.selectbox('Style',['white','whitegrid','dark','darkgrid'])

            r5=st.radio('Generate Plot',['n','y'])
            if r5=='y':
                sns.set_context("{}".format(context), rc={"font.size":9,"axes.titlesize":12,"axes.labelsize":14}) 
                sns.set_style("{}".format(set_style))
                s=sns.jointplot(data=data2,kind=kind,x=x,y=y,hue=hue,height=height,palette=palette) 

                try:
                    s.fig.suptitle(t='{} VS {}'.format(x,y).upper(),x=0.15,y=1.01,ha='left',va='baseline',fontweight='bold',fontsize=15)
                    with st.container():
                        fig=plt.savefig('seaborn.png',bbox_inches='tight',dpi=200)
                        st.pyplot(fig=plt)   
                        with open("seaborn.png", "rb") as file:
                             btn = st.download_button(label="Download image",data=file,file_name="seaborn.png",mime="image/png")
                except:
                    with st.container():
                        fig=plt.savefig('seaborn.png',bbox_inches='tight',dpi=200)
                        st.pyplot(fig=plt)   
                        with open("seaborn.png", "rb") as file:
                             btn = st.download_button(label="Download image",data=file,file_name="seaborn.png",mime="image/png")
        except:
               st.warning('Choose Something')
    if ctype=='Heat Map':   
        try:
            index=st.multiselect('Select Index',qual)
            columns=st.multiselect('Select Columns',qual)
            values=st.multiselect('Select Values',quant)
            aggfunc=st.selectbox('Select Aggregate Function',['count','min','max','mean','sum'])
            height= st.sidebar.slider('Height',1.0,20.0,8.0)
            width= st.sidebar.slider('Width',1.0,20.0,8.0)
            palette=st.sidebar.selectbox('Palette',['rocket','mako','crest','flare','magma','viridis','rocket_r','cubehelix','Blues','icefire','Spectral','coolwarm'])
            context=st.sidebar.selectbox('Context',['paper', 'notebook', 'talk', 'poster'])
            set_style=st.sidebar.selectbox('Style',['white','whitegrid','dark','darkgrid'])


            r6=st.radio('Generate Plot',['n','y'])
            if r6=='y':
                sns.set_context("{}".format(context), rc={"font.size":9,"axes.titlesize":12,"axes.labelsize":14}) 
                sns.set_style("{}".format(set_style))

                dpivot=data2.pivot_table(index=index,columns=columns,values=values,aggfunc=aggfunc)
                sns.set(rc = {'figure.figsize':(width,height)})
                s=sns.heatmap(dpivot,cmap=palette)
                if aggfunc=='min':
                    try:
                        plt.title('Minimum of {} for each {} vs {}'.format(''.join(',').join(values),''.join(',').join(columns),''.join(',').join(index)).upper(),fontsize=16, fontweight='bold')
                        plt.xlabel('{}-{}'.format(''.join(',').join(values),''.join(',').join(columns)))
                        with st.container():
                            fig=plt.savefig('seaborn.png',bbox_inches='tight',dpi=200)
                            st.pyplot(fig=plt)   
                            with open("seaborn.png", "rb") as file:
                                 btn = st.download_button(label="Download image",data=file,file_name="seaborn.png",mime="image/png")
                    except:
                        with st.container():
                            fig=plt.savefig('seaborn.png',bbox_inches='tight',dpi=200)
                            st.pyplot(fig=plt)   
                            with open("seaborn.png", "rb") as file:
                                 btn = st.download_button(label="Download image",data=file,file_name="seaborn.png",mime="image/png")
                if aggfunc=='max':
                    try:
                        plt.title('Maximum of {} for each {} vs {}'.format(''.join(',').join(values),''.join(',').join(columns),''.join(',').join(index)).upper(),fontsize=16, fontweight='bold')
                        plt.xlabel('{}-{}'.format(''.join(',').join(values),''.join(',').join(columns)))                        
                        with st.container():
                            fig=plt.savefig('seaborn.png',bbox_inches='tight',dpi=200)
                            st.pyplot(fig=plt)   
                            with open("seaborn.png", "rb") as file:
                                 btn = st.download_button(label="Download image",data=file,file_name="seaborn.png",mime="image/png")
                    except:
                        with st.container():
                            fig=plt.savefig('seaborn.png',bbox_inches='tight',dpi=200)
                            st.pyplot(fig=plt)   
                            with open("seaborn.png", "rb") as file:
                                 btn = st.download_button(label="Download image",data=file,file_name="seaborn.png",mime="image/png")
                if aggfunc=='mean':
                    try:
                        plt.title('Mean of {} for each {} vs {}'.format(''.join(',').join(values),''.join(',').join(columns),''.join(',').join(index)).upper(),fontsize=16, fontweight='bold')
                        plt.xlabel('{}-{}'.format(''.join(',').join(values),''.join(',').join(columns)))                        
                        with st.container():
                            fig=plt.savefig('seaborn.png',bbox_inches='tight',dpi=200)
                            st.pyplot(fig=plt)   
                            with open("seaborn.png", "rb") as file:
                                 btn = st.download_button(label="Download image",data=file,file_name="seaborn.png",mime="image/png")
                    except:
                        with st.container():
                            fig=plt.savefig('seaborn.png',bbox_inches='tight',dpi=200)
                            st.pyplot(fig=plt)   
                            with open("seaborn.png", "rb") as file:
                                 btn = st.download_button(label="Download image",data=file,file_name="seaborn.png",mime="image/png")
                if aggfunc=='sum':
                    try:
                        plt.title('Sum of {} for each {} vs {}'.format(''.join(',').join(values),''.join(',').join(columns),''.join(',').join(index)).upper(),fontsize=16, fontweight='bold')
                        plt.xlabel('{}-{}'.format(''.join(',').join(values),''.join(',').join(columns)))                        
                        with st.container():
                            fig=plt.savefig('seaborn.png',bbox_inches='tight',dpi=200)
                            st.pyplot(fig=plt)   
                            with open("seaborn.png", "rb") as file:
                                 btn = st.download_button(label="Download image",data=file,file_name="seaborn.png",mime="image/png")
                    except:
                        with st.container():
                            fig=plt.savefig('seaborn.png',bbox_inches='tight',dpi=200)
                            st.pyplot(fig=plt)   
                            with open("seaborn.png", "rb") as file:
                                 btn = st.download_button(label="Download image",data=file,file_name="seaborn.png",mime="image/png")
                if aggfunc=='count':
                    try:
                        plt.title('Count of {} for each {} vs {}'.format(''.join(',').join(values),''.join(',').join(columns),''.join(',').join(index)).upper(),fontsize=16, fontweight='bold')
                        plt.xlabel('{}-{}'.format(''.join(',').join(values),''.join(',').join(columns)))                        
                        with st.container():
                            fig=plt.savefig('seaborn.png',bbox_inches='tight',dpi=200)
                            st.pyplot(fig=plt)   
                            with open("seaborn.png", "rb") as file:
                                 btn = st.download_button(label="Download image",data=file,file_name="seaborn.png",mime="image/png")
                    except:
                        with st.container():
                            fig=plt.savefig('seaborn.png',bbox_inches='tight',dpi=200)
                            st.pyplot(fig=plt)   
                            with open("seaborn.png", "rb") as file:
                                 btn = st.download_button(label="Download image",data=file,file_name="seaborn.png",mime="image/png")
        except:
               st.warning('Choose Something')
    if ctype=='Correlation Map':   
        try:
            method=st.sidebar.selectbox('Method',['pearson', 'kendall', 'spearman'])
            cbar=st.sidebar.selectbox('Cbar',['y','n'])
            cbar_orientation=st.sidebar.selectbox('Cbar Orientation',['v','h'])
            height= st.sidebar.slider('Height',1.0,20.0,8.0)
            width= st.sidebar.slider('Width',1.0,20.0,8.0)
            rotationx= st.sidebar.slider('Rotate X Label',0,90,90)
            rotationy=st.sidebar.slider('Rotate Y Label',0,90,0)
            linewidths=st.sidebar.slider('Line Width',0.0,5.0,1.0)
            palette=st.sidebar.selectbox('Palette',['rocket','mako','crest','flare','magma','viridis','rocket_r','cubehelix','Blues','icefire','Spectral','coolwarm'])
            context=st.sidebar.selectbox('Context',['paper', 'notebook', 'talk', 'poster'])
            set_style=st.sidebar.selectbox('Style',['white','whitegrid','dark','darkgrid'])

            r7=st.radio('Generate Plot',['n','y'])
            if r7=='y':
                if cbar=='y':
                        if cbar_orientation=='h':
                                    sns.set(rc = {'figure.figsize':(width,height)})
                                    sns.set_context("{}".format(context), rc={"font.size":9,"axes.titlesize":12,"axes.labelsize":14}) 
                                    sns.set_style("{}".format(set_style))                                
                                    s=sns.heatmap(data2.corr(method='{}'.format(method)),cbar=True,cbar_kws={"orientation": "horizontal"},cmap=palette,linewidths=linewidths)
                                    plt.yticks(rotation=rotationy)
                                    plt.xticks(rotation=rotationx)
                                    plt.title('Correlation Heat Map'.upper(),fontsize=11, fontweight='bold')
                                    plt.tight_layout()
                        if cbar_orientation=='v':
                                    sns.set(rc = {'figure.figsize':(width,height)})
                                    sns.set_context("{}".format(context), rc={"font.size":9,"axes.titlesize":12,"axes.labelsize":14}) 
                                    sns.set_style("{}".format(set_style))
                                    s=sns.heatmap(data2.corr(method='{}'.format(method)),cbar=True,cbar_kws={"orientation": "vertical"},cmap=palette,linewidths=linewidths)
                                    plt.yticks(rotation=rotationy)
                                    plt.xticks(rotation=rotationx)
                                    plt.title('Correlation Heat Map'.upper(),fontsize=11, fontweight='bold')
                                    plt.tight_layout()
                if cbar=='n':
                            sns.set(rc = {'figure.figsize':(width,height)})
                            sns.set_context("{}".format(context), rc={"font.size":9,"axes.titlesize":12,"axes.labelsize":14}) 
                            sns.set_style("{}".format(set_style))
                            s=sns.heatmap(data2.corr(method='{}'.format(method)),cbar=False,cmap=palette,linewidths=linewidths)
                            plt.yticks(rotation=rotationy)
                            plt.xticks(rotation=rotationx)
                            plt.title('Correlation Heat Map'.upper(),fontsize=11, fontweight='bold')
                            plt.tight_layout()
                with st.container():
                        fig=plt.savefig('seaborn.png',bbox_inches='tight',dpi=200)
                        st.pyplot(fig=plt)   
                        with open("seaborn.png", "rb") as file:
                             btn = st.download_button(label="Download image",data=file,file_name="seaborn.png",mime="image/png")
        except:
               st.warning('Choose Something')
except:
    st.warning('Upload any csv/xlsx file')

st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')

st.write('### **About**')
st.info(
 """
            Created by:
            [Parthasarathy Ramamoorthy](https://www.linkedin.com/in/parthasarathyr97/) (Analytics Specialist @ Premium Peanut LLC)
        """)
