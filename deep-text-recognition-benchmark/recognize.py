# -*- coding:utf-8 -*-

import argparse
import string
import csv
import os
import pandas as pd
import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
from utils import CTCLabelConverter, AttnLabelConverter
from dataset import RawDataset, AlignCollate
from model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def demo(opt):
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    model.eval()
    str_list = list()
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index, preds_size)

            else:
                preds = model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)

            
#             log = open(f'./log_demo_result.txt', 'a', encoding='utf-8')
            log = open('{}.txt'.format(str(opt.out_csv_name).split('.csv')[0]), 'a', encoding='utf-8')

            dashed_line = '-' * 80
            head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'
            
            print(f'{dashed_line}\n{head}\n{dashed_line}')
            # log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                if pred_max_prob.cumprod(dim=0).nelement() == 0:
                    confidence_score = 0
                    pred = '###'
                else:
                    confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')
                log.write(f'{img_name:25s} {pred:25s} {confidence_score:0.4f}\n')
                str_list.append(pred)
                data = str(img_name).split('/')[1]
                img_counter = str(data).split('_')[0]
                x1 = str(data).split('_')[2]
                x2 = str(data).split('_')[3]
                y1 = str(data).split('_')[4]
                y2 = str(str(data).split('_')[5]).split('.jpg')[0]
                with open(opt.out_csv_name, 'a', encoding='utf-8', newline='') as public:
                    writer = csv.writer(public)
                    writer.writerow(['img_{}'.format(img_counter), str(x1), str(y1), str(x2), str(y1), str(x2), str(y2), str(x1), str(y2), pred])
            log.close()
    # return str_list[0]

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', default='recog/', help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
    parser.add_argument('--batch_size', type=int, default=3000, help='input batch size')
    parser.add_argument('--saved_model', required=True, default='saved_models\TPS-ResNet-BiLSTM-Attn-Seed1111(1130)/best_accuracy.pth', help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='漫髮作工室型娜安黛包熬早真田庄豆花堂食品臻珍永銀樓桃樂思便當專賣店柒壹契喫彩虹屋翰大里嘉榮物流件元壓油空管鋼膠高輝濂傢俱歐系德統巧三商福加站全國飛勒輪胎泰豐集團身廠中壢汽車電池聯所會合普書溫市立圖華館旱建農二理我可道交平路鐵門題問代事務營業項目有柵款貸借一主金不軍企勞保做承行信用良協助辦房土地間持分民繼設定轉移整資同配債融停心聽看小前方錄影監視牛排麵飲吃黑白切炒飯訂購話撞部世界漿球巨人明比醫診修改衣服長頸鹿和通租各牌機維聖往後火化下封閉數位財更到學眼科鏡街洋茶啡喝咖頭皮健豬菲琳攝林記萊爾富廣源水果鑫現削萬興子秤精批發湍駿帶藥西痛風鍍膜打蠟今局青圈族段台北緊急出口請勿在廂逗留滅器號歡魚玄雞橋梅慢尋九臭鍋再社區收送線乾洗名卡拉上啊囉哈湾之星延直瑞最好夾嫂佳香緻你天了嗎清祐鐘錶預約願采美容護膚東牙植腦鑲瓷特別引進重知魯肉古味炸骨滷唇創於年新屈臣氏臺消陽箱指外堡冠際吉祥老期照顧糕筒仔米晴川接待遠雄寶蓮扁翁家爌腳燉枸杞虱越南河粉友景燙菜歸鴨藤成自餐倆伊仕柏石網日深夜貨快潔鞋隨手關坪售迎沙床組墊每豪登嶺智匯城面央湯淺力告五王士男階場益暢障秀慕零誠龍威點式底摩按足展旭雅廳琦漾揚碩鴻料百音材響壽皇洲亞蓄互儲陳焿山禾博多院仁產其佐太兒嬰郵還園光晚康政穎劉江險陶種苗輛克吳竹秉宏限公司訊泡沫蛋海動客來注意左私費鉅放非本住戶尚時內芮曼瑪璉遵向循印象鎖澤何鋒島飾桌訓朋入暫券臨鬚張鬍堤丹傳剩餘嗽咳冒感治燒潤捲妝扒昌茂璽珠玉套具紗婚蒸腐烹板佑佛木匠的帽盛師楊計焜黃焙烘縫怡衛浴撒凱御邱傷術鮮養漢井岡阿義羊爐官存芳權量販窗簾施鬆依緹柳角章片表格千莉媽饅正草藝沖是愛曇跆拳涼雪厚吐申續攜碼趣順序經典夢駝鈴衝刺煎利危靠近捷運藏巖輕馨傘慶讚糖滿禮鼎廊姆等體驗生活昔野隱形挺灣使振蜜密邦賴模範濺夠弓納買兄弟蚵價襪折鹽原羅胖熗菓芝鄉莊藍森鹹都紫軒宴補習班證字第松神筆文純製造翅冰淇淋餅旺寵超飼得速旅爺霸爪選頑概念府巴妃法孔佰渡惠聚志庭殿昇鎮轅境頂廟拖脫桶把桿娃女汕刻勝君六桂宗親達檳榔咪李姐童裝弘騰遊戲欣賓首相功廈幼敏郭英檢想斯語劵蕉派盈素婦京壞狗宿技療性免開刀淨諾獸浮羹碗粿喜庫奇常雷根尺銷魂舖十震旦四棒驛酒娛忠哥為度而廚備翔烤村季守趴騎港顏月幸悅至居宣奶紅介軟泉極蓋究巷兩岸晶菱曆慈悲陀降善無蔬他櫃毒環蝦圓霖帳鎧省熊座丸煜培律婆圃坊宮估葵畫囈教瑤宇董鐲腊命擇釣灰程藕蔥抓像菁吧潮堆塔惟馬育械研宸焊晨茗弄魠汪勳丁徵壱番周杰倫應鄰谷演示然旁煮回標褒紋蒜唐薯腿塩鯖卷漏拆除勢防寰尖貿棉屬玻璃津版股份八饋双蘋樹微基處羽毛跨寫癒策提供講葛洛校蠣蛤氣担將倉偉搞烏辣妹湖迪雲朵澄億噌焢硯韓雜七嘴舌臘謝丞甜鱔疙瘩冷芋露傑午鋁採熱棚架欄杆汁如囍陸奈魏洪瑩燈鋪鈺炫瘋乖取鵝脆春蘇耳鼻喉彎窩粥耕饗泳懷瀧蔘麥芽韻俊妍扉肚纜埔優餃眾彌徠穩實鍠固拌騏泓被胤輇嵐煤臥吋盤殊苑拓帆布邊賢棧只制鳥誕祺櫥領域觀肆雕樺薪玩哩鶴杯丿虎仙碳複籠即暮復毅恆帝綵羿崑鎰媚聲圍棋異投幣甲馥麪級熔岩舞很忙呂假叉久恩初脯儷縣職苔巻淞淥耀禪寺右筠荺議埕賊謙姨唯榨檸檬腸恭姿液先劃赤崁由隆龜苓膏后礙坡禁致徐傅麗舍饌讓步秘訣曾椰夏暉塑總挽臉屏剪覺菇附綺紹杏尿彰妮蕾盟蒂酥宋綿榕汎菸頻調変渣绮濃乳它艾頓墨耗餡顆彬駛牆測游少伸擔嘿婷栗呷齁揪棻橙凍升紀譽孝員詩肯櫻濟群杜劇猴燦坤瑚舘刷曜止乙紙綠鯨授見坑嫩招啾嘣勇秋寢兔泫滋甘肺輔陰者亮慧麻波卍予參葆猛璾啟割厦蜂湘茄蕃宝亭嗲鈑邑繳鑽層羮母算獅考芒迷熟讀緑楨添鮑拇苙穀糧旗課錦瓜閔懶銘舒適睡眠砂編堃休閒辰翠翡啤脂琪浩享鬥汗綜靚寬葉賀串搜宵端薑涮絲蠶岱勁冊禎罷譏推築籟歲峰嵌褙裱框崚連賃芙貝值剌呈樣齡裕鴦鴛起播阪仟葳棠靖朝情媒干側爭飽銅黎滙气尼拿俗則構妶腎晟磅湃也澎銓勵儀索鈿易簧彈柚簡璿均駕綸稻控潞皓昶漠鍊鎂倍射斑扭瘦顯鉛色渼趙蘭攤蓬窯浯漁寿坐笑梁變廉伍糰韭受惜巢瀚眷嬛瑯雙短詠吊翼報沒凡您雨換列芬珈拍患嚴舉違妳去尤跌損針灸節試盒醬洽痘肌共駁乘两麟艦轟夫擂斜探啦莎蒙册註與耐滑質案楽類酬爽賽菌殺氧嶼蹟伴糬姜范靈泥漆減剋尊兆拋睿歌睫蕙委孩鱻丼丙次郎喏贈督歇煲瑜驊昱碟輯瓏玲單~貞盅柯奧過低酮增就癮遇終艷潢帘壁絕獨芭搬沅緣締禦牧擴伽擀橘溢斌醎始柔閣誼鳳箋劑血捐酷戌渥詢諮娫歆著需錢促完蒞棟舊憶插捧盆釀枝塊醒礦喚能戀碰巾踏宙強菩薩因任責競週獲頒貢獻獎梨炭透奨孫皆父以洱琺努爹珂朗夥哭欠酸鳴喇叭敬及僅孕凸栓染刨半梯拜籍狂餛醋祖燥飩披對苦傻砰鈦那輸洸粘魷州乃搖吮潛艇淬勤艋舺紡織楓瓦蔡詮卓岳刈噴擎鍵呼蜀綫盞茸燕惡魔殼胡希望託又己蔗擊導糊船玥贊姊榭宰疾病貓貴細練荳紳操畜嘜軋螺憩鬘玟麒崙堅碧粽脾笙洏托椒溪釦劍殷煙壺煌晉伙呱默膳焦朱芯叫奢若乍散弱矯寓帥宥齒时貼絡態釋衡穴蕊濕腹腔暖鈞焗嗑饈符杓蒔離喬调变廷璞崗規穆榛箴言鄧甄充貫茴禧螢幕搶霜頁硬們顎湧伝污廬茅另含娘咕嚕粹隊伯磨蹈繆彭姬醇厲害抽個齊仲彥薇瑛磐瞇沛耶穌條幻頌滄悦紓赫沏蜓渴貳拾玖差析蘆跑哦齋撈找載茵郁燴莒闔扇炙要疫鬼刃反斗濰墘嫣繡弁額兵昆儘剔挑檀磁磚宜胆袋毯恒戰航飄剉席遷瓶濠丈賦必鉑戒負偕汶羚麝募澡廿琴肝炎呵查肢陣癌篩抹胃吸喘臟陞廖邸瑾嫁逢剎鼠鈔票蘿暐鵬薄檜氛裁狀般怩聿候滾蕭奕霞値靜云靓哇蕯評論嚐僑喆沐塗琧晏俠碁誰峻憲冬妙刑警察簽証稅咨付掃描例息町玫瑰從当宅寮饕賞逸茱嘗幫浦罐仿潭葱驕緯嘟蟹闆給驚龐曲窈窕忱效率洰撥筋佩叮噹背嗜猶未盡穗偲救涎凹洞淡抗皺菠羨祕頤饀飆垣爵蓁粄怪琇占卜弗霧眉支撐走膝腰跟傾伶黏袖坷塏鷹凌恋纖裂絨蔔蝶酪映紉瞳奴戚爸蕎裡皂沃覽認才逰飪裏陵鎗叢結涵繪亦掛岐椎肥繕弦曦曙武窮隔桔敦翻醉筍曉翊說胥酌霏聊饡暘址嫚丰解決笛叔吼鍾鋸儒此褲撼抄瑋韋蓉哪蕬炉妻匋豊鋐窖蔓価电号体噞殝侓粧籤鰭爬許喔萱刮痧什麼畢菘煉盧朧厝綉莫梵誌坦銳仂婕姻佈煞瑟荷燭湄貮爆聰淳副馳懸迴丟棄鹼罩罰垃圾扶遛繫援填臼肩挫賈蒲鰻柬刊灶倢併悠郡忍或姚焱处滇蛻叄噶忌珊餞捕捉鮪榜隻錨鰲売滴渾霓淑雯堯浙庚奏瓣鯉錫難众从烈妤退暑諭醣睛煦骼漂餚燻藹芸佶裹于淼準寄斐间莘溯攀崇瀝彫魅渝叨轄積璇詹吿寂迫崎烊杵汀捏祝炖頰薦夷凰淘莓鼓靑穿腩膽戴翎画迦怕曬墾欲骰埋釘凝閑識閱崴俏紐澳召鄭檯觸剛丘抱陪樸篇繁犬梭会酵蒼仰故昏綱疼叁喵躍賠較趕薬寡椪嶄拷匙遙搾令鵲鶕葚欽潰瘍腺悍汙辨掏朶垂窄蕁膀恁燿卉棕束籃栽置擺噸掠奪粱禿嚨隹狼瘡癬溼疹槍煥犀破轎這係学熏脚囝阮獄逃鴉姓狐狸已握艙拔礎殖柴准核é宓錡署夯麴畔磺訪薈顫吅憑斷儕峨嵋忘嘻失症疲妏彼鍚晃駐墅鍰惦兜閃滝葷沢鷄寨嚼餌碑弥枇杷帖抵菈沉腓敷杉鮭鱈唱皰痣疤樑錠瓊愣咘謀腫瘤菖—牲扯竿暴佬紮征峯桐祭橫嗚倪蛙亀釜録鱘杭碎尬鉄珮略尾求炮盥荒蔴悰混槽奔隍夲鑼遣落磡歷史孟熙革袁酉確鴿咔灘啓箭摃累悉討营联娟距攻嘆椅筌痠驅邪錚倒悔叻蟲阻氐塲琚閥缸捞媞帛亨蠔疆狠激銨遺肋徒咸粗拘鍛兌犇執勰撕胸眸豔舫蜊妥迅擘哆揭誓緬甸溜韮朴儂叡茟灌拼伏佢掌冶茉晋詳覆逾踨追並須謹寒汰哲襯鑑蹦衫殘灒箔曹点旬哉菊薛黨汝奉哨痔祿聘夭埼塵缺濱邁亂滬畏懼芊亢砧皿橄欖銲麋卵燊桑舜輿堪佔悟聆縮禍轆漸裳堀軌尉紘羞嚇翹泌返雀豋余償戽塢姑裙着獵軸吾摸鏢馜胞嫻絜旨芛諧措亥媳蜡璟潘牽划歯榴疊枕蚊庠兼滌舶卸靄擁敲覓甕饞餓蟬吞脊氹沁萃恰琍乎徽虔錯燁肴鈕皈豚豫砍极騋瓢禹斤搭幾閎国偶寧攏袂愈怹譜捌截璀璨浜蹄鯛瞻叟欺楷卦劈籽遜霍爛薺虫悶臍鏽樽鏟畝溝荖寅瑈跳蟳袓葡薡粒旋垠恐慌慮鬱咾乓乒胱肽竣澢烽芹癡舟聞娮礼椿萄処璜饒魁茹彙懂鮨劣塹橡条榆歧娥昭暟泵沺弼暗沈旻琲妞鍗韌幟坎庵戊梧脹屑掉辛廁岑竟圗梓浸鑛僮檣婰繞尙龔繃丫线糞允緩璋廢牠疏铜壯泊糟柱賜饃璸哞鋰昕繩禕嬌鉉团駭捆艸匾鑄綋薏粑梗替関稔筷盜裴妯娌鯤玳蟑稼柑荃筑撰朕避鉤嬤珺籐竺豹庤芥磊驒據杍澱篝摔喳蹤矽韵樟嗓弋鶯醡膩帕师隠蔵訴甫峽隙槌閩蔣敵逕諒暨麺崧役娶訶諦僧爍爱沸薰僖昀塾罈刁晒椗夕浪堉耙祈蔚抑秒鱗慾摺饐潑餵甩瑨鄒吔綢糙縱胚涂昂卅擦恤傭甁稀贗擱脈蔻盔蝴聶搗薌湛竭衰粵萌敘諸亷斬蛇唷鴆奮桴寸檔鏈碞瀊卿棗惑誘諺葫藻屯灃巃瀑巡遮蓆喪䥶惍稚烀蟎譚荏苛錵姸淮烙谈蚤末猜昜啉玹虛咩靶徑鯰酢琩茨玺猫峇祠壷鷲鈣炊泛審萳恕躲籌函緞呢羢沾盎鯊盲坣礫抿勃跡萫悸逆屁琉孜絮甭顴嘎吟拒炳押丽醜囋餽筵颺斧鋤薔尗泷衙摯牢鑰弍灑膺稽扣盾醐醍甦侯幹竤贏痰瀉冲崔衆恬锅坮宬銹猿粳璦謬哺泗懿慚愧秦酩醺犁几恪歉咱啥咚刚楀喧嘩啇囯万抙逛賔渦肖碇罕邀麩噘夀晰説災鋭喱鏵袍彤搵歪篤婭剃粩埆偏籬摘敢琼談鐿鉦孚栩肪揉侵氫侖粟葯狄蒟蒻唸慨腋抖綻途侑厭賒淵隼棨楠貌趾屆擬瑠惶胜萍蝨囊婉咬鱉矮閹螃孰巒渠熄湞筴卒紛繽塘埤砥謐泱遼愉漲槤菡旌侶滲塞壅遨壬箏挖税禱压淅弎啞竇暈眩蔭湳丨瓔困矞彗厘悌咒勘稱痕秝蘊焼殯葬羴滕株猪扳炁扎懋毓騼醮壇癸凉妖傍髓踢踐蘑燜苟雖畊奎尭剝槓傲啄遞玬礴撿馮濁碾俄絓竂丐壕鱸虞盃魟俐汐药椏氟呆噲塚禽湲釔舵', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    parser.add_argument('--out_csv_name', type=str, required=True, help='path to save the pred result')
    parser.add_argument('--label_root', type=str, required=True, help='path of the pred label of yolo')
    parser.add_argument('--img_path', type=str, required=True, help='path of the image (public or private or others)')
    
    opt = parser.parse_args()

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character += string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    # public img path
    imgPath = opt.img_path

    # predict result path   
    rootPath = opt.label_root
    labelPath = rootPath+"labels/"
    
    # result list, which save all the predict result (string)
    result = []
    
    # read the csv
    for i in range(21000, 25500):
        print('{}img_{}.jpg'.format(imgPath, i))
        img = cv2.imread('{}img_{}.jpg'.format(imgPath, i))
        imgH = img.shape[0]
        imgW = img.shape[1]
        print('{}img_{}.txt'.format(labelPath, i))
        if os.path.isfile('{}img_{}.txt'.format(labelPath, i)) == False: 
            continue
        f = open('{}img_{}.txt'.format(labelPath, i),'r')
        count0, count1 = 0, 0
        for line in f.readlines():
            data_split = line.split(' ')
            class_id = int(data_split[0])
            c_x = float(data_split[1]); c_y = float(data_split[2]); b_width = float(data_split[3]); b_height = float(data_split[4])
            x1 = int((c_x-(b_width/2)) * imgW); x2 = int((c_x+(b_width/2)) * imgW); y1 = int((c_y-(b_height/2)) * imgH); y2 = int((c_y+(b_height/2)) * imgH)
            x1 = max(x1, 0); y1 = max(y1, 0); x2 = min(imgW-1, x2); y2 = min(imgH-1, y2)
            crop = img[y1:y2, x1:x2]
            if class_id == 1:
                count1 += 1
                cv2.imwrite('pred1/{}_{}_{}_{}_{}_{}.jpg'.format(i, count1, x1, x2, y1, y2), crop)
            else: 
                count0 += 1
                cv2.imwrite('pred0/{}_{}_{}_{}_{}_{}.jpg'.format(i, count0, x1, x2, y1, y2), crop)
    
    
    opt.imgH = 64; opt.imgW = 64
    opt.saved_model = "saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/best_accuracy.pth"
    opt.character = '際旅行社遠生醫院永祥佛帽工藝觀音的店壹酒肆金軒國華神像雕刻榮發俱百貨上品堂中診所咖啡深藏名代服飾巧欣皮鞋老泰羊肉夠土司樺薪補習班成功書局時在好型玩頭髮經典哩鶴九製麵鹿杯飲食兩丿線小虎牛排售冰封仙果雪花世家影印都碳烤紅樓館依加複合式手籠湯包早到晚汽車宏全實業勝文理立兒美沙龍利達幸福苑奇即期商菲比英暮流整復菜根香素天鵝空壓機千新清潔有限公德鴻不動產捷保養廠冠毅冷氣東購物心恆大帝鐘錶眼鏡台南企股份綵羿體容歸仁自助餐坊崑鎰平價媚幼園育作室聲電程嘉霖智圍棋星樂教滷聯軍異投幣洗衣鮮芋莉水界胖豆甲竹創意豬料豐源味馥漫麪蕉派超級巨正牙熔岩燒舞居皇器釣魚子遊場蔬很忙呂記鴨邦銀方莊渡假飯鋪愛慶灣第一匠茗三叉路檳榔健宸和漿久興藥房茶聖恩話鐵初原向陽賴棧傳承古道瑞士珠寶劉日脯今悅來務儷宴婚會唐風縣裝運職男震旦通訊歡精燉海苔巻淞淥客元廚吉漢堡壢牌耀嶺學圓光禪寺請右轉真筠緹顏預約專荺鑫益高雄議鹽埕街賊仔市四集炸晴康順謙楊志明板埔阿姨唯珍榨取檸檬腸油年川吃恭喜財左萬里虹蔘姿批月昇液晶視維修續尚豪陶屋先鋒配件劃建赤崁由窗片優活緻傢火腿隆龜苓膏后鬆餅無障礙坡禁誠地政事忠骨王致徐師傅人麗舍各種零饌角尖夾娃泓辦處信用菁語秘訣讓進步分農村區屈臣氏玉山嵐拉號駿曾青草椰賣禾夏隱形安暉鄉下黑帆布輝具免北塑膠總匯魯傘泉昌麥饅井脆雅如挽臉屏現煎卡多斯剪燙湍覺菇附設拆綺數位哥民族何紹耳鼻喉科杏林彩券焿做臘快便當選奶粉尿長照彰化鉅城妮蕾晨盟親俊傑蒂酥雞出租婦停宋河綿榕樹我寵乾速相汎示展菸頻調変炒渣打绮微技筆腦双營涼濃乳它壞艾普頓盛張芳協墨材耗表械倉餡顆五律彬鎮駛牆面告測前游勿淨桃定輪後少伸擔響開喝嘿熊臭媽鍋義婷苗洋栗二億呷齁揪力偉棻足鼎燈橙揚帳凍備景而升紀譽西孝買廣員詩肯櫻瑩眾濟群馬杜港石滿劇外戶團頂春猴燦坤借款證田丹瑚舘版刷曜止入乙吧碗紙善慈特葛娜綠計妝造藍鯨授富見坑嫩招咪啾嘣境網蔥勇秋黃妍琳曼納粥旺登寢童兔象本泫蒸滋陳甘肺輔陰治者亮慧周島麻舖之湾十波卍予餃木蛋參葆膚松簾猛璾七澤啟李官臺森隨翔切割厦蜂湘檢驗基茄蕃宝烘焙亭郵嗲蚵嫂鈑邑段六研威慢宮間延蓄儲你為費繳京鑽站往鬚鬍八層饗韓羮點陀母蘋黛同想算獅友考法芒堤迷可熟讀緑毛販浮午楨添鮑秀校拇頸苙術穀雜糧旗課錦純閔密瓜哈煮懶得圈蓮廳佳銘舒適睡眠床住砂險介統編堃休閒辰熱翠翡啤越萊脂攝球環私琪浩資顧問享芝鬥思汗身部綜格靚寬倫葉賀串夢墊搜尋宵夜亞鳥端雲霸薑涮絲蠶岱碼尺府凱撒勁冊禎歐罷別譏推築茂籟施歲防消峰嵌鑲琦璃玻指摩騰按褙裱框畫碩旭崚廊連賃諾芙貝怡值米目剌套穎呈樣翅煤齡裕鴦辣鴛播每起直阪梅仟葳棠靖朝情巴津帶宣腳等媒干庭側爭賓盤銅飽黎滙气概念植尼勢朵陸拿俗療則然放騎江拓構妶腎晟急貿銷秤磅知糖軟其湃也澎鋼重內收丁弘銓杰仕紗馨勵劵壽儀惠索鈿祐盈易白簧彈柚簡胎璿均駕訓綸稻制振控輛護潞焊皓昶非漠吐鍊鎂倍底臻勞佑域毒射注除斑雷扭瘦顯鉛色渼趙蘭攤蓬鹹窯浯漁輕寿坐回笑梁變廉伍良門伊架朋糰韭受勳口感惜巢瀚近眷嬛瑯蒜雙短詠拖管吊紫窩翼桂逗報宗鱔沒凡您痛鎖權雨更換折系列芬珈拍患嚴舉違糕淋淇巷尤妳去跌損傷針灸櫃節送培烏試迎盒禮拌醬洽刺痘肌首共接駁乘两麟艦轟夫擂斜探啦野莎蒙標册註與爺鋁耐使潤滑質案楽庄汁類酬爽廈看賽菌殺氧捲耕嶼蹟伴糬圖池倆橋姜范靈性泥漆減溫量剋漏尊臨兆拋女改睿兄錄監厚爾歌睫彌蕙央委吳孩領鱻範模潮丙丼次申郎腐爌喏沖贈督歇煲瑜驊昱碟輯瓏玲單貞盅太甜瑪柯奧過低酮增就癮遇終擇艷潢帘壁櫥絕讚獨芭搬祺焢沅緣締禦訂是牧主題擴伽擀湖妹橘溢斌醎始柔閣誼澄婆鳳箋劑殿血捐酷季戌渥慕詢諮娫歆應只著需錢妃佰促谷克完蒞芮棟危舊御憶字插捧盆釀枝塊箱璽醒礦喚能泳戀碰撞座桌巾踏宙強菩薩鄰因任責競週頒獲獸項獎獻貢究狗閉謝坪飛郭梨囍炭透奨鈴孫皆父以洱最關琺努爹存柏珂朗宇徵夥哭欠酸鳴喇叭他敬緊及供僅孕凸栓染刨餘梯半拜籍番鍍膜狂邊餛嘴醋祖燥飩披對苦傻宿砰貸嬰交廟鈦削組採那輸洸粘羹魷州乃泡喫搖吮潛艇淬勤魂艋舺紡織楓瓦蔡詮卓岳再刈蠟噴引擎留章丸鍵呼蜀綫盞茸燕董漾魠惡魔殼胡懷希望託度又己蔗棒擊導糊蜜船玥贊姊榭蓋宰疾病貓貴細練荳紳浴操畜君肚嘜軋螺殊提憩鬘玟麒極崙堅碧瑤粽脾菓笙洏托椒溪釦常翰刀劍殷煙瓷壺煌晉伙羅娛戲洪呱繼默膳焦朱聽芯叫奢若乍散弱矯寓帥宥时齒貼絡態蕊濕腹腔暖癒釋衡穴鈞焗嗑饈符杓蒔循離喬调变廷璞崗規瘋穆騏榛箴言鄧甄充貫茴禧螢幕搶霜頁硬們顎湧伝污廬茅省虱另含娘咕嚕粹隊伯至磨蹈繆趣彭姬醇羽爪厲害暢抽個齊蛤仲彥薇瑛塔庫磐瞇沛聚耶穌條幻頌滄悦赫紓沏蜓渴貳拾玖差析蘆洲跑哦齋撈找靠博載茵郁燴莒闔扇炙疫守要鬼滅刃反斗濰墘嫣紋繡債弁額兵昆儘剔挑檀磁磚宜胆袋岸毯爐腊采恒衛於戰航飄剉嗎席了遷移瓶濠丈賦必鉑屬戒負粿玄筒偕洛迪汶羚麝募階澡廿琴呵查寫肢陣璉癌篩抹胃吸冒喘炎肝臟敏陞廖邸瑾嫁逢剎鼠鈔票蘿暐鵬薄檜氛裁丞狀般怩聿候滾蕭奕霞衝値靜云靓哇蕯評論嚐僑喆沐塗琧穩晏塩碁誰俠峻憲固冬蝦妙刑警察簽証稅咨付掃描例息町玫瑰從当宅寮邱饕待巖賞逸茱嘗幫浦韻罐仿潭葱驕緯嘟蟹闆還給驚龐曲窈窕忱效率互洰撥筋佩叮噹背嗜猶未盡穗偲救涎凹洞淡抗皺菠羨祕頤饀飆演垣爵廂蓁粄疙瘩怪持琇弄占卜弗圃霧眉弓支撐走膝腰跟傾伶黏袖坷塏策鷹凌恋纖裂絨蔔蝶酪映縫紉瞳奴戚爸蕎裡皂沃覽認才逰烹飪裏炫旁陵勒鎗叢結抓涵唇繪亦掛岐椎肥繕弦曦曙武窮隔桔敦佐翻薯醉筍曉翊說胥酌霏聊饡暘講卷址嫚搞丰解決笛叔吼鍾飼鋸儒此棉褲撼抄將瑋韋蓉哪蕬炉妻匋豊鋐窖蔓価电号体噞殝侓粧籤鰭爬許弟喔萱刮痧什麼瀧畢菘姐煉盧朧厝綉莫梵誌坦銳仂婕命姻佈煞瑟荷燭湄貮灰攜爆聰淳脫副馳懸迴丟棄鹼罩垃圾罰扶遛繫援填臼肩挫賈奈蒲鰻柬刊灶倢併悠郡忍或姚焱处滇蛻叄噶忌珊餞捕捉鮪榜隻錨鰲売滴魏渾霓淑賢雯堯浙庚奏瓣鯉昔錫舌挺難众从烈襪妤退暑諭醣睛煦骼漂餚燻藹芸佶裹于淼準寄斐间莘露溯攀崇瀝彫魅噌渝叨轄積璇詹吿寂被迫崎烊駝杵汀捏祝炖頰薦夷凰淘莓柳鼓葵靑穿腩膽戴翎熬画迦怕曬墾欲骰埋釘凝閑識閱崴俏紐蘇澳饋召鄭檯觸剛估跨丘抱陪樸篇犬梭繁会酵蒼仰翁故昏綱疼叁臥喵躍賠較趕薬寡椪嶄拷匙遙搾令鵲鶕葚欽潰瘍腺悍汙芽桶辨掏朶鈺垂窄蕁膀恁燿卉棕束籃栽置擺噸掠奪粱禿孔嚨隹狼瘡癬溼疹槍煥菱犀破轎這係学熏脚囝願阮獄逃鴉姓狐狸已拳把握吋艙拔礎殖柴准核宓堆誕錡署夯麴畔磺訪薈融顫吅憑斷儕嵋峨忘嘻失症疲妏彼鍚晃駐墅鍰惦兜閃滝葷沢鷄寨嚼餌碑弥枇杷帖抵菈沉腓敷杉鮭鱈唱皰痣疤樑錠瓊愣咘謀腫瘤菖牲扯竿暴佬紮征峯桐祭橫嗚鐲倪蛙亀釜録鱘淺杭碎尬鉄珮略尾求炮盥荒蔴悰扁混槽奔隍夲鑼遣落磡藕歷史孟熙革袁酉確寰鴿咔灘啓箭摃累悉討营联娟惟距攻嘆椅筌痠驅邪錚倒悔序叻蟲阻氐塲琚岡閥缸捞媞帛亨蠔疆狠激銨遺肋曆徒咸粗拘鍛兌犇執勰撕胸眸豔舫蜊妥迅擘哆揭誓緬甸溜韮汕朴儂叡茟灌拼伏佢掌冶茉晋詳覆逾踨追並須謹寒汰哲襯咳嗽鑑蹦衫殘灒箔曹点藤旬哉菊薛黨汝奉哨痔祿聘夭埼塵缺濱邁亂滬畏懼芊亢砧皿橄欖銲麋卵燊桑舜輿堪佔悟聆縮禍轆漸裳堀軌尉紘羞嚇翹泌返雀豋余償囈戽塢姑裙着獵軸吾摸鏢馜胞嫻絜旨芛欄諧措亥媳蜡璟潘牽划歯榴疊枕蚊庠兼滌舶桿卸靄擁敲覓甕饞餓蟬吞扒彎脊氹沁萃恰琍乎徽虔錯燁肴鈕皈豚豫砍蠣极騋瓢禹斤搭幾閎国偶寧攏袂愈怹譜捌截璀璨浜〇蹄鯛瞻叟欺楷卦劈籽遜姆霍爛薺虫悶臍鏽樽鏟畝溝荖寅瑈担跳蟳袓葡薡粒旋垠恐慌慮鬱咾乓乒胱肽竣澢烽鯖芹癡舟聞娮礼椿萄処璜沫饒魁茹彙懂鮨劣塹橡条榆歧娥昭暟泵沺弼暗沈壱旻琲妞鍗韌幟坎庵戊梧脹屑掉辛廁岑竟圗梓浸鑛僮檣婰繞尙龔繃丫线糞允緩璋廢牠疏铜壯泊糟柱賜驛饃璸哞鋰昕繩禕嬌鉉团駭捆艸匾鑄綋薏粑梗替関稔筷盜裴妯娌鯤玳蟑稼柑荃筑撰朕避鉤嬤珺籐竺豹庤芥磊驒據杍澱篝摔喳蹤矽韵囉樟嗓跆弋鶯醡膩帕师隠蔵訴甫峽隙槌杆閩蔣敵逕諒暨麺崧役娶柒訶諦僧爍爱沸薰僖昀塾煜罈刁晒椗夕浪堉耙祈蔚抑秒鱗慾摺饐潑餵甩瑨鄒吔綢糙縱胚涂昂卅擦恤傭甁稀贗熗擱脈蔻盔蝴聶搗薌湛竭衰粵萌敘諸亷斬蛇唷鴆奮桴寸檔鏈碞瀊卿棗惑誘棚諺葫藻屯灃巃瀑巡遮蓆喪䥶惍稚烀蟎譚荏苛濂錵姸淮烙谈蚤末猜昜啉玹鎧虛咩靶徑鯰酢硯琩茨玺猫峇祠壷鷲鈣趴炊泛審萳恕躲籌函緞呢羢沾盎鯊盲坣礫抿汪勃跡萫悸逆屁琉孜絮甭顴嘎吟拒炳押丽醜囋餽筵颺斧鋤薔尗泷衙摯牢鑰弍灑膺稽扣盾醐醍甦侯幹竤贏痰瀉冲崔契衆恬锅坮秉宬銹猿粳璦謬哺泗懿慚愧秦酩醺犁几恪歉咱啥咚刚楀喧嘩啇囯万抙逛賔渦肖碇罕邀麩噘夀晰説災鋭喱鏵袍彤搵歪篤婭剃暫粩埆偏枸籬摘敢琼談鐿轅鉦孚栩肪揉侵氫侖粟葯狄蒟蒻唸慨遵腋抖濺綻途侑厭賒淵隼棨楠貌趾屆擬瑠惶胜萍蝨囊婉咬纜鱉矮閹螃孰巒渠熄湞筴卒紛繽塘埤砥謐泱遼愉漲槤菡旌侶滲塞壅遨壬箏挖税降禱压淅弎剩啞竇暈眩蔭湳丨瓔困矞彗厘悌咒勘稱痕秝蘊焼殯葬羴滕株猪扳炁扎懋毓騼醮壇癸凉妖傍髓踢踐蘑燜苟雖畊奎尭剝槓傲啄遞玬礴徠撿馮濁碾俄絓竂丐壕鱸虞盃魟俐汐药椏氟呆噲塚禽湲釔舵酱鸭传这话：郑总闯涯虾，鱼孙记锡纸饺她们饭题厅馄饨庆连锁贩贵阳农汤圆乐惹馆盖浇兴业书桥烫检兰剁馅啵腾讯应鲜滚给来汉订肘黄焖鸡诚铺绵经济实绿庐忆芦红创种疯价带请楼动嗦风货铭厨卖嘞个尝浆®馋匆储耿强夹馍陕过门奥广场怀旧设菏泽签撩为龙网铁统鱿闻见热卤选还衷乡烧镜优、斓机头华蝎发梦区·领楚车视荐杞缘萝犟轰阁岛纯专长萨饮东挞装哒荣厂怼纤户独宫鹅（）员资议儿岭鹏欢舰铃饼戈牟厠陈苕启择烟现进惊绝浏刘态达鳗—厉蚝韩净钱额杂粮筐运转务｜缤灿腕试减问内询。樐开渔约劳佘轻艺觅钵马宽潼关尔滨让样综肠财贡煨档单县鲁并临簋晗幢购皖块颜¥瘾跷边质赵烩吹闲简撸据鸽贝亲纪吕汇无寻斋护鸣臊丝顺围蛳庙缃▪呀补阴润类较些畅脉帮对杨轩脑项许宾买赠笃尽™识栖凤柠图杀归宁赣粤盘颓医张满槑级馕验昙罗圣ǐà隐状饱费语乌齐闸飞养邻笼银吴炝瑢莲哎麦泾瓯∣歺玛远温溏飘劲楸姝苏与莱错诗脖丛の乖别咥辽贴办属°岁园摆_坚鸳鸯饪么￥弹库谜答节《瀞严憨镇酿抢突观澜谁爷码欧标谭衢际饿荟组说顾锦嗨鲍胗栈葑谊够绪“”编谢屿钟队觉没幺鲩噜牦义驾逍遥鹃妈稣耍岚缆顿圭蚁蚂矿呗钝摄窝荞搅陇狮晖箐针蚬雒←复柜驴证账汆稍戏菋卫匹栋邢莞篷剡帅维笔历适倾潜沧颈训邵诺职诊扬褚糁钢盐绍槿逼兽闺贺甏碚饰帐飮仓随拴桩贱贾层潍叠积纷苹珀琥龄饹饸郞贤嫡监汍喻骄评侠赢馒产导倡丧确蓝岗札树响赖驻辉盗软纱环坛协负啊赛奋叶涛铝鞭俞愚奚浚宠炲横沥称彪诉屠鲅荤载－雾厢苼祛堰爅叼鲈栏…挂报胶幂吻聪糯糍烓鸟亚肤滩荆绚黔纳课贸鄢贯宛Ⅱ凯闽纬瀘琅琊夺扩诱浓郊羔涩捻羯励嵊墙眯预释逐灵勺岂俵将圳伦漓仅谦硕抚郝计杖陆娅翟邮•笨拨烂馈溶销难噢摊弯换搓勾缅啪吓涨妆吗幅术杈泼郫誉膨灯踪细舔愿衹婴邕浔闪襄【丢赔鳍ā濤兑币盱眙搏燃叙腌闫芈谣矾锣跃钥驼禄轮认荀死尧届获崂顶柿臂凭慵懒静淌甚绣渌艳¦揽沂听舆谐疗菟贞胪鉴衔蝉芜稳豌亿缙沤迟忧伟晓枫窑祚鸿烦恼闖潯测荔巫Λ暹兮似离继责啖赞焰乔拱骚扰喷驢仨豉邳终扫恶争‘肃绕贼笋钩诸榧骆访递藩篓莅但覃沟楹洒抛险损蹭萧颗剂抻锋权蒝侬芷庞毕谱祉樱甑→綦躁涧馀潇须纺织军砀麯剑娱链炼锤献謎数则邓赐猩厌沪沌腻熳Áō専卢酆缠汊濮唝坂莴楂蒋伞瑧冈讲详鹵摇偃嵩￣乞谨′剥颐▕？捅缕艰参䬺趁呦饵垦袜熠显缇寇槟絳驿歹耘竞­唛习涡鄂蘸佼ǒ∶裤寳嚸鲢俤∧坟废挪気辘碉雁捣烁Ⓡ脸樊玮★扑渊醴瑶霊呜～备钻冻菽❤咻籣尹岔壮霄浒娇傣熘孤邂逅厕郸莆虽倦怠矣俩录驰续钦党鲫侧叕谛峡⑧澧异朔冯战决棍；﹣丑妇焉坞壳馐帜娄鳯秜结稠齿矫恵缔皲渭鹤蛎鞍赴盼鳌络陂濑馇孃规论凳贰兎阅读◆练墩↓栢枣罉鳅顽倌曰颖巍沽荘啃髙○且阆匣阜遗仑昊构挡巳镖燚栾啫饷韶菒仪脏浅辈靡埠溧睦举钳哟扞侣龚龟钜埭が搽螞蟻娚垵☎骑Ξ谋黍侍赏扮洁闹鶏φ麓喂Ⓑ奖砖仉腱朙莹抺该穇箩嬷廰祗坝蛮槁亖俯浣锨闷梆▫姥烎枪羌穷晕箕婧辅柃「」≡ɔ唔盏睇淖沣礁豇栙涌钓迁妨仍漳鳕拐餸绘漕餠áǎ邨闵锹兹咀篮势卟频甬卞憾駱涉珥熹萦瑭谓兢攸䒩柗鲤粕汾怎ī盯氽镀∙纵逝姣旮旯︳屉㕔а铸韦击伿『』☆声牵夸愁噻嗏嵗湶営汴咑㸆嗖敌炕锌堽邯珑尕喃脱邹烨檐碌页荠囬肫淀鳝鲺龘｝｛犹坏烜媛桓堇↑扛罄鲶鍕携卑遍蓓坨馔姫堵烛阔噤湿鳜崽＋灼腥峭喽郏栀鲨诈斥螄稿掂鹭乱埃禅鹌鹑û囧阡眞嘱鳞浃贷荊讨陌冀砵鵺鹰啜咽鯡蒌蒿③嬢瓮嚣莽咋饥凈隧鉢茫íóΘ祎褥梳湫︵︶伤榄拽舂斩飨执渤惺箬茜俭瀏嫦琶琵咿吖戛舱祁缝羋绒淝锐嗯睁＇既虚娴碱哚逻匪ü槐写踩赚觞鲽闰璐侨煸棵峪π粙喊饶孵屎陦莳倔祸灮莜淩鲮き缪糠埧痴猎嬉凼舅褐醪氷碛绽瀾孖雍òǘ涤辜嘢┐徫ステキ迈虢糐挥纹枊劝俺粢馓拥嘶ㄍ丩ㄝホルモニ颂噫否绎垒锄卧炯轴隴宀荥贪麽暧恨礳叽断岢睾迹禺沭瀛唤愤怒郷狱蠡咶垅绅馏襠郴恺樵呐砸鄞叹遂吆嘛灭蜗泄鑪淄钰壶旱蕴峧赶咏渎靣趟镫讷迭彝却稷庖扉瞿筏韧墟俚翕貂搄寞泸炘茯骏糘毫幽ぉにぃ炽斛窜鲷爲裸窦羡冕粶䬴嚟辆撮隋赁咭崟沩颠诠拙瞄漖橱帼崮勋苍雏睐袭皋陝彻垚咯凑纽巩茏昧坯霾闳凖皱缗箍筹孬唠输驭哼匡偵壩蝇贛漟邴謠怿亁棱阶堔炜笠遏犯罪仝珅咧摑滘颁锈佤佗卌匱藺蔺塍鯽鳟畿耦吨䒕茬枼桼嘍沱楞屹掺挢荻偷辶泮某聂甡吁鎬谅鞘泪鐡犊涪杬睢しんいちょくどぅ戍莼蒡砣惯隣撇筛昵涞绥俑鐉埒侗仗违辫灞琰崖炑昝迩浈挚砚ú缴黒畈忽燍姗逹岫践嗞㥁祯牤诏杠苞滤鞑萤榶轨耒嚮漪键彦词敖鸦秧囚绾镶帷豁煒珲緋仆ラナン瀨缓氵汥殡靳闭偘酔哑佚1234567890abcdefghijklmnopqrstuvwxyz'
    opt.image_folder = 'pred0/'
    demo(opt) 
    
    opt.imgH = 32; opt.imgW = 100
    opt.character = string.printable[:-6]
    opt.saved_model = "modules/TPS-ResNet-BiLSTM-Attn-case-sensitive.pth"
    opt.image_folder = 'pred1/'
    demo(opt) 