import urllib.request
import urllib.parse
import json
import hashlib
from google_trans_new import google_translator
import similarity as s



#语种
#法语 fra
#西班牙语 spa
#日语 jp
#俄语 ru
#韩语 kor
#德语 de
#英语 en
#中文 zh


# 可更改的有翻译的内容、语种
def translate_baidu(en_str):
    URL='http://api.fanyi.baidu.com/api/trans/vip/translate'
    From_Data={}  #创建From_Data字典，存储向服务器发送的data
    From_Data['from']='zh' #输入的语种
    From_Data['to']='en' #翻译的语种
    From_Data['q']=en_str     #要翻译的数据
    From_Data['appid']='20210113000670420'       #申请的APPID
    From_Data['salt']='1435660288'        #随机数
    Key='UK9njd0CMJn7xNPbzLT4'                    #平台分配的密匙
    m=From_Data['appid']+en_str+From_Data['salt']+Key
    m_MD5=hashlib.md5(m.encode('utf8'))
    From_Data['sign']=m_MD5.hexdigest()

    data=urllib.parse.urlencode(From_Data).encode('utf-8')  #使用urlencode()方法转换标准格式
    response=urllib.request.urlopen(URL,data)            #传递request对象和转换完格式的数据
    html=response.read().decode('utf-8')          #读取信息并解码
    translate_results=json.loads(html)            #使用JSON
    #print(translate_results)                      #打印出JSON数据
    translate_results=translate_results['trans_result'][0]['dst']   #找到翻译结果

    print('翻译的结果是: %s'%translate_results)               #打印翻译信息

def translate_youdao(content):
    url = 'http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule'
    data = {}
    data['i'] = content #翻译的内容
    data['doctype'] = 'json'  
    data['from'] = 'zh' #翻译前的语种 
    data['to'] = 'en'    #翻译后的语种
    data = urllib.parse.urlencode(data).encode('utf-8')
    response = urllib.request.urlopen(url,data)
    html = response.read().decode('utf-8')
    target = json.loads(html)
    result = target['translateResult'][0][0]['tgt'] #找到翻译结果
    print("result: %s" %result)

def translate_google(content):
    translator = google_translator()
    From_Data={}  #创建From_Data字典，存储向服务器发送的data
    From_Data['from']='zh' #输入的语种
    From_Data['to']='en' #翻译的语种
    text = translator.translate(content,lang_src=From_Data['from'],lang_tgt=From_Data['to'])
    print(text)

    
if __name__ =='__main__':
    content = "我喜欢你"
    translate_baidu(content)
    translate_youdao(content)
    translate_google(content)
    # 有毒吧你
    # sb文宝


