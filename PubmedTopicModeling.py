import numpy as np
import pandas as pd
import sys

from Bio import Entrez, Medline
import json
with open('entrezemail.json', 'r') as f:
    data = json.load(f)
    Entrez.email = data['email']  # CONTACT EMAIL NECESSARY FOR ENTREZ ACCESS

import matplotlib
import matplotlib.pyplot as plt
plt.close("all")
matplotlib.style.use('ggplot')

import nltk
import gensim
import stop_words

import networkx as nx



def searchpubmed(searchquery="cardiomyocytes, calcium", returnmax=20):
    print('Search Pubmed')
    searchqueryinput = input("Search query: (default: cardiomyocytes, calcium)")
    if searchqueryinput:
        searchquery = searchqueryinput
    handle = Entrez.esearch(db="pubmed", term=searchquery, retmax=returnmax) #retmax is 20 by esearch default
    queryresponse = Entrez.read(handle)
    handle.close()
    print("\n" + queryresponse['Count'] + " results available for \'" + str(searchquery) + "\', returning " + str(returnmax))
    queryids = (searchquery, queryresponse['IdList'])
    return queryids

def fetchrecord(inputids, numberofrecs, queryinput):
    print('Downloading Pubmed Records...')
    handle = Entrez.efetch("pubmed", id=str(inputids), rettype="medline", retmode="text")
    records = Medline.parse(handle)
    records = list(records) 
    
    recordsdf = pd.DataFrame(records)    
    print('\nFeatures available: {}'.format(recordsdf.columns.values.tolist()))
    recordsdf['Searched'] = str(queryinput)        
    reckeys = ['PMID', 'TI', 'AB', 'DP', 'PHST', 'Searched']
    print("\nReturning {} features : {} ".format((len(reckeys)-1), reckeys[0:(len(reckeys)-1)]))   
    recordsselectdf = recordsdf[reckeys]
    
    recordsdf.to_csv('full_records' + str(numberofrecs) + '.csv')    
    recordsselectdf.to_csv('selected_records' + str(numberofrecs) + '.csv')
    
    handle.close()
    return recordsselectdf 

def getdata(numberofrecs):
    print('Checking Archives...')
    try:
        readrecordsdf = pd.read_csv(('selected_records' + str(numberofrecs) + '.csv'), index_col=0)  
        queryinput = readrecordsdf['Searched'][0]
        print('{} records retrieved for \'{}\''.format(numberofrecs, queryinput)) 
    except OSError:
        if int(numberofrecs) < 10001:
            [queryinput, idlist] = searchpubmed(returnmax=numberofrecs)
            readrecordsdf = fetchrecord(idlist, numberofrecs, queryinput)
        else:
           print('please check size and format restrictions')
           sys.exit()
    return {'SearchQuery':queryinput, 'RecordsDataFrame':readrecordsdf}

def cleandata(inputresultdf):          
    outputresultdf = inputresultdf.copy()  
    print('\nFormatting Data...')
    outputresultdf.PMID = pd.to_numeric(outputresultdf.PMID, errors='coerce')
    outputresultdf.DP = pd.to_datetime(outputresultdf.DP, errors='coerce')    
    outputresultdf = outputresultdf[outputresultdf.TI.notnull()]
    outputresultdf = outputresultdf[outputresultdf.AB.notnull()]
    print('Extrapolating Missing Publication Dates...')    
    NewDate = outputresultdf.loc[:, 'DP'].copy() 
    for row in outputresultdf.itertuples():
        if type(row.DP) is pd.tslib.NaTType:
            if (type(row.PHST) is not float):
                takedatestring = str(outputresultdf.PHST[row.Index]) 
                NewDate[row.Index] = pd.to_datetime(takedatestring.split("'")[-2].split(" ")[0])
    outputresultdf['ReleaseDate'] = NewDate
    outputresultdf = outputresultdf[outputresultdf.ReleaseDate.notnull()]
    print('Returning Cleaned Data...')
    return outputresultdf

def tokenizefortopicmodel(inputseries, stopwords=stop_words.get_stop_words('en')):
    print('\nTokenizing {} Records...'.format(inputseries.name))
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')   
    tokenstemmer = nltk.stem.regexp.RegexpStemmer('ies$|ing$|s$|ia$|ic$|is$|ly$|y$|able$', min=2)
    
    texts = []
    inputlist = inputseries.values.astype(str).tolist() 
    for i in inputlist:
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
        stemmed_tokens = [tokenstemmer.stem(i) for i in tokens]
        stopped_tokens = [i for i in stemmed_tokens if not i in stopwords]
        longer_tokens = list(filter(lambda x: ((len(x) > 1) and x != 'nan'), stopped_tokens))
        texts.append(longer_tokens)
    return texts 

def topicmodel(trainingtexts, testtexts, numtopics=5, numwords=3):
    print('\n\nDiscovering {} topics'.format(int(numtopics)))
    dictionary = gensim.corpora.Dictionary(trainingtexts)
    trainingcorpus = [dictionary.doc2bow(text) for text in trainingtexts] # document - term matrix
    testcorpus = [dictionary.doc2bow(text) for text in testtexts] # document - term matrix

    print('Computing Document Topics...')
    lda = gensim.models.LdaModel(trainingcorpus, id2word=dictionary, num_topics=numtopics, passes=10, \
        minimum_probability=0.01)

    print('\nTopics\' Terms')
    topics_termids = []    
    topics_termwords = []  
    topics_terms = []
    for i in range(int(numtopics)):    
        topics_termids.append(lda.get_topic_terms(i, topn=10)) # list tuples (termid, probability), top n terms per topic          
        topics_termwords.append(lda.show_topic(i, topn=10))    # list tuples (term, probability), top n terms per topic
        terms = []
        for term in topics_termwords[i]:
            terms.append(term[0])
        topics_terms.append(terms)  # list top n terms per topic
        
    topictermsdf = pd.DataFrame()
    topictermsdf['TopicTermIDsWeighted'] = topics_termids
    topictermsdf['TopicTermsWeighted'] = topics_termwords
    topictermsdf['TopicTerms'] = topics_terms
        
    simi = gensim.similarities.MatrixSimilarity(lda[testcorpus])

    print('Documents\' Topics')
    docs_topics = []
    docs_maintopics = []    
    similarity_matrix = pd.DataFrame()
    for i in range(len(testtexts)):
        listtopntopics = sorted(lda[testcorpus[i]], key=lambda x: x[1], reverse=True)   
        docs_topics.append(listtopntopics) #Ranked topics per doc, tuples with probability, probability > minimum 0.01 by default
        try:
            docs_maintopics.append(listtopntopics[0][0]) # top topic
        except:
            print('Try reducing \'minimum_probability\' threshold in LDA topic model')
        
        sims = simi[listtopntopics]
        pd.DataFrame(sims)
        similarity_matrix[i] = sims  


    return {'Dictionary':dictionary.id2token, \
    'TopicTermsDF':topictermsdf, \
    'DocTopProbAbs':docs_topics, \
    'DocTopProbAbsHighest':docs_maintopics, \
    'SimilarityMatrix':similarity_matrix \
    } 
    
   
def topicnetwork(inputarray, numtops):
    plt.figure()
    topiclist=[]
    for i in range(int(numtops)):
        topiclist.append('Topic ' + str(i))                
    G=nx.Graph()
    G.add_nodes_from(topiclist) 
    for i in range(int(numtops)):
        getstarseries=[]
        getstarseries.append(topiclist[i])
        for term in inputarray['TopicTerms'][i]:
            getstarseries.append(term)
        G.add_star(getstarseries)
    nx.draw_networkx(G, node_size=10, edge_color='r', font_color='k', font_weight='bold') 
    plt.title('Topic Term Distribution Network')
    plt.show()    
    
def pubyearhistogram(pubdate, topicplotted='cardiomyocyte'):
    print('\nPlotting \'{}\' Publication Frequency by Year...'.format(topicplotted))
    try:
        plt.figure()
        histpubdate = pubdate.ReleaseDate.dt.year
        mintime = min(histpubdate)
        maxtime = max(histpubdate)
        totallength = len(histpubdate)
        print("Publication dates for {} records ranging within {} years, {} to {}".format(totallength, maxtime - mintime, mintime, maxtime))
        binwidth = 1
        sidespace = 5
        histpubdate.plot.hist(bins=np.arange(mintime, maxtime + binwidth, binwidth), alpha=0.5)
        plt.xlim(mintime - sidespace, maxtime + sidespace)
        plt.title("{} Recent Publications on: {}".format(totallength, topicplotted))
        plt.xlabel('Publication Year') 
        plt.grid(True)
        plt.show()
    except:
        plt.close()
        try:
            print('No histogram for {}, all records published in {}'.format(topicplotted, max(topicdocdf.ReleaseDate.dt.year)))        
        except:
            print('Subthreshold Topic')
    
def topicdistributionhistogram(topicarray, wholearray):
    with plt.style.context('bmh'):
        plt.figure()        
        mintime = min(wholearray.ReleaseDate.dt.year)
        maxtime = max(wholearray.ReleaseDate.dt.year)
        totallength = len(wholearray.ReleaseDate.dt.year)
        print("\nPlotting Topic Distribution by Year...")
        print("Topic disribution and frequency per year, for {} records ranging within {} years, {} to {}".format(totallength, (maxtime-mintime), mintime, maxtime))
        binwidth = 1
        sidespace = 5
        plt.hot()
        plt.hist(topicarray, histtype='barstacked', bins=np.arange(mintime, maxtime + binwidth, binwidth), alpha=0.5, label=((range(int(numsearchtopics)))))
        plt.xlim((mintime - sidespace), (maxtime + sidespace))
        plt.title("Topic Distribution in {} Recent Publications on: \'{}\'".format(totallength, querysearched))
        plt.xlabel('Publication Year')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.legend(loc=0, title='Topic', fontsize='small')
        plt.show()







######## EXECUTE ######## 

if __name__ == "__main__":
    ## Get Data
    numberofrecs = input("How many listings should we download (100 < recommended < 10,000)? \n") 
    numsearchtopics = min(max(10, round(int(numberofrecs)/200, -1)), 30)  # limits for num topics
    print('\nSeek {} topics in {} documents'.format(int(numsearchtopics), int(numberofrecs)))   
    
    getdatadict = getdata(numberofrecs)
    querysearched = getdatadict['SearchQuery']
    getresultdf = getdatadict['RecordsDataFrame']
    cleanedresultdf = cleandata(getresultdf)
    
    ##Topic Model:
    customstopwords = stop_words.get_stop_words('en', cache=False)
    newstopwords = ['effect', 'increased', 'increase', 'decreased', 'decrease', 'inhibit', 'result', \
    'role', 'regulate', 'via', 'associated', 'associate', 'new', 'inhibitor', 'antagonist', 'agonist', \
    'dependent', 'independent']
    for term in newstopwords:
        customstopwords.append(term)
    for term in querysearched.split(", "):
        customstopwords.append(term)

    cleanedresultdf['TokensTitles'] = tokenizefortopicmodel(cleanedresultdf.TI, stopwords=customstopwords)
    cleanedresultdf['TokensAbstracts'] = tokenizefortopicmodel(cleanedresultdf.AB, stopwords=customstopwords)
        
    topicmodeldict = topicmodel(cleanedresultdf.TokensTitles, cleanedresultdf.TokensAbstracts, numtopics=numsearchtopics, numwords=10)
    term_dictionary = topicmodeldict['Dictionary']
    topic_terms_df = topicmodeldict['TopicTermsDF']    
    doc_similarity_matrix = topicmodeldict['SimilarityMatrix']
    
    cleanedresultdf['DocTopics'] = topicmodeldict['DocTopProbAbs']   # topics per doc, tuples (topic, probability), ranked, for topics > minimum probability
    cleanedresultdf['DocTopicsTop'] = topicmodeldict['DocTopProbAbsHighest']  # top topic per doc
    
    ## Inspect    
    print(topic_terms_df['TopicTerms']) 
    topicnetwork(topic_terms_df, numsearchtopics)    

    pubyearhistogram(cleanedresultdf, topicplotted=querysearched)
    
    listoftopicdfs = []
    for t in range(int(numsearchtopics)):
        topicdocdf = cleanedresultdf[cleanedresultdf.DocTopicsTop == t]
        pubyearhistogram(topicdocdf, "Topic " + str(t))
        listoftopicdfs.append(topicdocdf.ReleaseDate.dt.year)

    topicdistributionhistogram(listoftopicdfs, cleanedresultdf)
    
    
    
    print('\nDone')