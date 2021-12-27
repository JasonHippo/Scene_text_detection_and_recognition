# -*- coding:utf-8 -*-
import os
import sys
import time
import random
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np

from utils import CTCLabelConverter, CTCLabelConverterForBaiduWarpctc, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model
from test import validation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(opt):
    """ dataset preparation """
    if not opt.data_filtering_off:
        print('Filtering the images containing characters which are not in opt.character')
        print('Filtering the images whose label is longer than opt.batch_max_length')
        # see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L130

    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    train_dataset = Batch_Balanced_Dataset(opt)

    log = open(f'./saved_models/{opt.exp_name}/log_dataset.txt', 'a', encoding="utf-8")
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size,
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)
    log.write(valid_dataset_log)
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()
    
    """ model configuration """
    if 'CTC' in opt.Prediction:
        if opt.baiduCTC:
            converter = CTCLabelConverterForBaiduWarpctc(opt.character)
        else:
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

    # weight initialization
    for name, param in model.named_parameters():
        if 'localization_fc2' in name:
            print(f'Skip {name} as it is already initialized')
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:  # for batchnorm.
            if 'weight' in name:
                param.data.fill_(1)
            continue

    # data parallel for multi-GPU
    model = torch.nn.DataParallel(model).to(device)
    model.train()
    if opt.saved_model != '':
        print(f'loading pretrained model from {opt.saved_model}')
        if opt.FT:
            model.load_state_dict(torch.load(opt.saved_model), strict=False)
        else:
            model.load_state_dict(torch.load(opt.saved_model))
    print("Model:")
    print(model)

    """ setup loss """
    if 'CTC' in opt.Prediction:
        if opt.baiduCTC:
            # need to install warpctc. see our guideline.
            from warpctc_pytorch import CTCLoss 
            criterion = CTCLoss()
        else:
            criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
    # loss averager
    loss_avg = Averager()

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))
    # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

    # setup optimizer
    if opt.adam:
        optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    print("Optimizer:")
    print(optimizer)

    """ final options """
    # print(opt)
    with open(f'./saved_models/{opt.exp_name}/opt.txt', 'a', encoding="utf-8") as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)

    """ start training """
    start_iter = 0
    if opt.saved_model != '':
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            print(f'continue to train, start_iter: {start_iter}')
        except:
            pass

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    iteration = start_iter

    while(True):
        # train part
        image_tensors, labels = train_dataset.get_batch()
        image = image_tensors.to(device)
        text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
        batch_size = image.size(0)

        if 'CTC' in opt.Prediction:
            preds = model(image, text)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            if opt.baiduCTC:
                preds = preds.permute(1, 0, 2)  # to use CTCLoss format
                cost = criterion(preds, text, preds_size, length) / batch_size
            else:
                preds = preds.log_softmax(2).permute(1, 0, 2)
                cost = criterion(preds, text, preds_size, length)

        else:
            preds = model(image, text[:, :-1])  # align with Attention.forward
            target = text[:, 1:]  # without [GO] Symbol
            cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

        model.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
        optimizer.step()

        loss_avg.add(cost)

        # validation part
        if (iteration + 1) % opt.valInterval == 0 or iteration == 0: # To see training progress, we also conduct validation when 'iteration == 0' 
            elapsed_time = time.time() - start_time
            # for log
            with open(f'./saved_models/{opt.exp_name}/log_train.txt', 'a', encoding="utf-8") as log:
                model.eval()
                with torch.no_grad():
                    valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels, infer_time, length_of_data = validation(
                        model, criterion, valid_loader, converter, opt)
                model.train()

                # training loss and validation loss
                elapsed_time /= 3600
                expect_total_time_cost = elapsed_time * opt.num_iter / (iteration+1)
                loss_log = f'[{iteration+1}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f} hours, Expect total time cost: {expect_total_time_cost:0.5f} hours'
                loss_avg.reset()

                current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.2f}'

                # keep best accuracy model (on valid dataset)
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_accuracy.pth')
                if current_norm_ED > best_norm_ED:
                    best_norm_ED = current_norm_ED
                    torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_norm_ED.pth')
                best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.2f}'

                loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                print(loss_model_log)
                log.write(loss_model_log + '\n')

                # show some predicted results
                dashed_line = '-' * 80
                head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                for gt, pred, confidence in zip(labels[:5], preds[:5], confidence_score[:5]):
                    if 'Attn' in opt.Prediction:
                        gt = gt[:gt.find('[s]')]
                        pred = pred[:pred.find('[s]')]

                    predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                predicted_result_log += f'{dashed_line}'
                print(predicted_result_log)
                log.write(predicted_result_log + '\n')

        # save model per 1e+5 iter.
        if (iteration + 1) % 1e+5 == 0:
            torch.save(
                model.state_dict(), f'./saved_models/{opt.exp_name}/iter_{iteration+1}.pth')

        if (iteration + 1) == opt.num_iter:
            print('end the training')
            sys.exit()
        iteration += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', help='Where to store logs and models')
    parser.add_argument('--train_data', required=True, help='path to training dataset')
    parser.add_argument('--valid_data', required=True, help='path to validation dataset')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--batch_size', type=int, default=1200, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=60000, help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=2000, help='Interval between each validation')
    parser.add_argument('--saved_model', default='', help="path to model to continue training")
    parser.add_argument('--FT', action='store_true', help='whether to do fine-tuning')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
    parser.add_argument('--baiduCTC', action='store_true', help='for data_filtering_off mode')
    """ Data processing """
    parser.add_argument('--select_data', type=str, default='/',
                        help='select training data (default is MJ-ST, which means MJ and ST used as training data)')
    parser.add_argument('--batch_ratio', type=str, default='1',
                        help='assign ratio for each selected data in the batch')
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                        help='total data usage ratio, this ratio is multiplied to total number of data.')
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=64, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=64, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str,
                        default='際旅行社遠生醫院永祥佛帽工藝觀音的店壹酒肆金軒國華神像雕刻榮發俱百貨上品堂中診所咖啡深藏名代服飾巧欣皮鞋老泰羊肉夠土司樺薪補習班成功書局時在好型玩頭髮經典哩鶴九製麵鹿杯飲食兩丿線小虎牛排售冰封仙果雪花世家影印都碳烤紅樓館依加複合式手籠湯包早到晚汽車宏全實業勝文理立兒美沙龍利達幸福苑奇即期商菲比英暮流整復菜根香素天鵝空壓機千新清潔有限公德鴻不動產捷保養廠冠毅冷氣東購物心恆大帝鐘錶眼鏡台南企股份綵羿體容歸仁自助餐坊崑鎰平價媚幼園育作室聲電程嘉霖智圍棋星樂教滷聯軍異投幣洗衣鮮芋莉水界胖豆甲竹創意豬料豐源味馥漫麪蕉派超級巨正牙熔岩燒舞居皇器釣魚子遊場蔬很忙呂記鴨邦銀方莊渡假飯鋪愛慶灣第一匠茗三叉路檳榔健宸和漿久興藥房茶聖恩話鐵初原向陽賴棧傳承古道瑞士珠寶劉日脯今悅來務儷宴婚會唐風縣裝運職男震旦通訊歡精燉海苔巻淞淥客元廚吉漢堡壢牌耀嶺學圓光禪寺請右轉真筠緹顏預約專荺鑫益高雄議鹽埕街賊仔市四集炸晴康順謙楊志明板埔阿姨唯珍榨取檸檬腸油年川吃恭喜財左萬里虹蔘姿批月昇液晶視維修續尚豪陶屋先鋒配件劃建赤崁由窗片優活緻傢火腿隆龜苓膏后鬆餅無障礙坡禁誠地政事忠骨王致徐師傅人麗舍各種零饌角尖夾娃泓辦處信用菁語秘訣讓進步分農村區屈臣氏玉山嵐拉號駿曾青草椰賣禾夏隱形安暉鄉下黑帆布輝具免北塑膠總匯魯傘泉昌麥饅井脆雅如挽臉屏現煎卡多斯剪燙湍覺菇附設拆綺數位哥民族何紹耳鼻喉科杏林彩券焿做臘快便當選奶粉尿長照彰化鉅城妮蕾晨盟親俊傑蒂酥雞出租婦停宋河綿榕樹我寵乾速相汎示展菸頻調変炒渣打绮微技筆腦双營涼濃乳它壞艾普頓盛張芳協墨材耗表械倉餡顆五律彬鎮駛牆面告測前游勿淨桃定輪後少伸擔響開喝嘿熊臭媽鍋義婷苗洋栗二億呷齁揪力偉棻足鼎燈橙揚帳凍備景而升紀譽西孝買廣員詩肯櫻瑩眾濟群馬杜港石滿劇外戶團頂春猴燦坤借款證田丹瑚舘版刷曜止入乙吧碗紙善慈特葛娜綠計妝造藍鯨授富見坑嫩招咪啾嘣境網蔥勇秋黃妍琳曼納粥旺登寢童兔象本泫蒸滋陳甘肺輔陰治者亮慧周島麻舖之湾十波卍予餃木蛋參葆膚松簾猛璾七澤啟李官臺森隨翔切割厦蜂湘檢驗基茄蕃宝烘焙亭郵嗲蚵嫂鈑邑段六研威慢宮間延蓄儲你為費繳京鑽站往鬚鬍八層饗韓羮點陀母蘋黛同想算獅友考法芒堤迷可熟讀緑毛販浮午楨添鮑秀校拇頸苙術穀雜糧旗課錦純閔密瓜哈煮懶得圈蓮廳佳銘舒適睡眠床住砂險介統編堃休閒辰熱翠翡啤越萊脂攝球環私琪浩資顧問享芝鬥思汗身部綜格靚寬倫葉賀串夢墊搜尋宵夜亞鳥端雲霸薑涮絲蠶岱碼尺府凱撒勁冊禎歐罷別譏推築茂籟施歲防消峰嵌鑲琦璃玻指摩騰按褙裱框畫碩旭崚廊連賃諾芙貝怡值米目剌套穎呈樣翅煤齡裕鴦辣鴛播每起直阪梅仟葳棠靖朝情巴津帶宣腳等媒干庭側爭賓盤銅飽黎滙气概念植尼勢朵陸拿俗療則然放騎江拓構妶腎晟急貿銷秤磅知糖軟其湃也澎鋼重內收丁弘銓杰仕紗馨勵劵壽儀惠索鈿祐盈易白簧彈柚簡胎璿均駕訓綸稻制振控輛護潞焊皓昶非漠吐鍊鎂倍底臻勞佑域毒射注除斑雷扭瘦顯鉛色渼趙蘭攤蓬鹹窯浯漁輕寿坐回笑梁變廉伍良門伊架朋糰韭受勳口感惜巢瀚近眷嬛瑯蒜雙短詠拖管吊紫窩翼桂逗報宗鱔沒凡您痛鎖權雨更換折系列芬珈拍患嚴舉違糕淋淇巷尤妳去跌損傷針灸櫃節送培烏試迎盒禮拌醬洽刺痘肌首共接駁乘两麟艦轟夫擂斜探啦野莎蒙標册註與爺鋁耐使潤滑質案楽庄汁類酬爽廈看賽菌殺氧捲耕嶼蹟伴糬圖池倆橋姜范靈性泥漆減溫量剋漏尊臨兆拋女改睿兄錄監厚爾歌睫彌蕙央委吳孩領鱻範模潮丙丼次申郎腐爌喏沖贈督歇煲瑜驊昱碟輯瓏玲單貞盅太甜瑪柯奧過低酮增就癮遇終擇艷潢帘壁櫥絕讚獨芭搬祺焢沅緣締禦訂是牧主題擴伽擀湖妹橘溢斌醎始柔閣誼澄婆鳳箋劑殿血捐酷季戌渥慕詢諮娫歆應只著需錢妃佰促谷克完蒞芮棟危舊御憶字插捧盆釀枝塊箱璽醒礦喚能泳戀碰撞座桌巾踏宙強菩薩鄰因任責競週頒獲獸項獎獻貢究狗閉謝坪飛郭梨囍炭透奨鈴孫皆父以洱最關琺努爹存柏珂朗宇徵夥哭欠酸鳴喇叭他敬緊及供僅孕凸栓染刨餘梯半拜籍番鍍膜狂邊餛嘴醋祖燥飩披對苦傻宿砰貸嬰交廟鈦削組採那輸洸粘羹魷州乃泡喫搖吮潛艇淬勤魂艋舺紡織楓瓦蔡詮卓岳再刈蠟噴引擎留章丸鍵呼蜀綫盞茸燕董漾魠惡魔殼胡懷希望託度又己蔗棒擊導糊蜜船玥贊姊榭蓋宰疾病貓貴細練荳紳浴操畜君肚嘜軋螺殊提憩鬘玟麒極崙堅碧瑤粽脾菓笙洏托椒溪釦常翰刀劍殷煙瓷壺煌晉伙羅娛戲洪呱繼默膳焦朱聽芯叫奢若乍散弱矯寓帥宥时齒貼絡態蕊濕腹腔暖癒釋衡穴鈞焗嗑饈符杓蒔循離喬调变廷璞崗規瘋穆騏榛箴言鄧甄充貫茴禧螢幕搶霜頁硬們顎湧伝污廬茅省虱另含娘咕嚕粹隊伯至磨蹈繆趣彭姬醇羽爪厲害暢抽個齊蛤仲彥薇瑛塔庫磐瞇沛聚耶穌條幻頌滄悦赫紓沏蜓渴貳拾玖差析蘆洲跑哦齋撈找靠博載茵郁燴莒闔扇炙疫守要鬼滅刃反斗濰墘嫣紋繡債弁額兵昆儘剔挑檀磁磚宜胆袋岸毯爐腊采恒衛於戰航飄剉嗎席了遷移瓶濠丈賦必鉑屬戒負粿玄筒偕洛迪汶羚麝募階澡廿琴呵查寫肢陣璉癌篩抹胃吸冒喘炎肝臟敏陞廖邸瑾嫁逢剎鼠鈔票蘿暐鵬薄檜氛裁丞狀般怩聿候滾蕭奕霞衝値靜云靓哇蕯評論嚐僑喆沐塗琧穩晏塩碁誰俠峻憲固冬蝦妙刑警察簽証稅咨付掃描例息町玫瑰從当宅寮邱饕待巖賞逸茱嘗幫浦韻罐仿潭葱驕緯嘟蟹闆還給驚龐曲窈窕忱效率互洰撥筋佩叮噹背嗜猶未盡穗偲救涎凹洞淡抗皺菠羨祕頤饀飆演垣爵廂蓁粄疙瘩怪持琇弄占卜弗圃霧眉弓支撐走膝腰跟傾伶黏袖坷塏策鷹凌恋纖裂絨蔔蝶酪映縫紉瞳奴戚爸蕎裡皂沃覽認才逰烹飪裏炫旁陵勒鎗叢結抓涵唇繪亦掛岐椎肥繕弦曦曙武窮隔桔敦佐翻薯醉筍曉翊說胥酌霏聊饡暘講卷址嫚搞丰解決笛叔吼鍾飼鋸儒此棉褲撼抄將瑋韋蓉哪蕬炉妻匋豊鋐窖蔓価电号体噞殝侓粧籤鰭爬許弟喔萱刮痧什麼瀧畢菘姐煉盧朧厝綉莫梵誌坦銳仂婕命姻佈煞瑟荷燭湄貮灰攜爆聰淳脫副馳懸迴丟棄鹼罩垃圾罰扶遛繫援填臼肩挫賈奈蒲鰻柬刊灶倢併悠郡忍或姚焱处滇蛻叄噶忌珊餞捕捉鮪榜隻錨鰲売滴魏渾霓淑賢雯堯浙庚奏瓣鯉昔錫舌挺難众从烈襪妤退暑諭醣睛煦骼漂餚燻藹芸佶裹于淼準寄斐间莘露溯攀崇瀝彫魅噌渝叨轄積璇詹吿寂被迫崎烊駝杵汀捏祝炖頰薦夷凰淘莓柳鼓葵靑穿腩膽戴翎熬画迦怕曬墾欲骰埋釘凝閑識閱崴俏紐蘇澳饋召鄭檯觸剛估跨丘抱陪樸篇犬梭繁会酵蒼仰翁故昏綱疼叁臥喵躍賠較趕薬寡椪嶄拷匙遙搾令鵲鶕葚欽潰瘍腺悍汙芽桶辨掏朶鈺垂窄蕁膀恁燿卉棕束籃栽置擺噸掠奪粱禿孔嚨隹狼瘡癬溼疹槍煥菱犀破轎這係学熏脚囝願阮獄逃鴉姓狐狸已拳把握吋艙拔礎殖柴准核宓堆誕錡署夯麴畔磺訪薈融顫吅憑斷儕嵋峨忘嘻失症疲妏彼鍚晃駐墅鍰惦兜閃滝葷沢鷄寨嚼餌碑弥枇杷帖抵菈沉腓敷杉鮭鱈唱皰痣疤樑錠瓊愣咘謀腫瘤菖牲扯竿暴佬紮征峯桐祭橫嗚鐲倪蛙亀釜録鱘淺杭碎尬鉄珮略尾求炮盥荒蔴悰扁混槽奔隍夲鑼遣落磡藕歷史孟熙革袁酉確寰鴿咔灘啓箭摃累悉討营联娟惟距攻嘆椅筌痠驅邪錚倒悔序叻蟲阻氐塲琚岡閥缸捞媞帛亨蠔疆狠激銨遺肋曆徒咸粗拘鍛兌犇執勰撕胸眸豔舫蜊妥迅擘哆揭誓緬甸溜韮汕朴儂叡茟灌拼伏佢掌冶茉晋詳覆逾踨追並須謹寒汰哲襯咳嗽鑑蹦衫殘灒箔曹点藤旬哉菊薛黨汝奉哨痔祿聘夭埼塵缺濱邁亂滬畏懼芊亢砧皿橄欖銲麋卵燊桑舜輿堪佔悟聆縮禍轆漸裳堀軌尉紘羞嚇翹泌返雀豋余償囈戽塢姑裙着獵軸吾摸鏢馜胞嫻絜旨芛欄諧措亥媳蜡璟潘牽划歯榴疊枕蚊庠兼滌舶桿卸靄擁敲覓甕饞餓蟬吞扒彎脊氹沁萃恰琍乎徽虔錯燁肴鈕皈豚豫砍蠣极騋瓢禹斤搭幾閎国偶寧攏袂愈怹譜捌截璀璨浜〇蹄鯛瞻叟欺楷卦劈籽遜姆霍爛薺虫悶臍鏽樽鏟畝溝荖寅瑈担跳蟳袓葡薡粒旋垠恐慌慮鬱咾乓乒胱肽竣澢烽鯖芹癡舟聞娮礼椿萄処璜沫饒魁茹彙懂鮨劣塹橡条榆歧娥昭暟泵沺弼暗沈壱旻琲妞鍗韌幟坎庵戊梧脹屑掉辛廁岑竟圗梓浸鑛僮檣婰繞尙龔繃丫线糞允緩璋廢牠疏铜壯泊糟柱賜驛饃璸哞鋰昕繩禕嬌鉉团駭捆艸匾鑄綋薏粑梗替関稔筷盜裴妯娌鯤玳蟑稼柑荃筑撰朕避鉤嬤珺籐竺豹庤芥磊驒據杍澱篝摔喳蹤矽韵囉樟嗓跆弋鶯醡膩帕师隠蔵訴甫峽隙槌杆閩蔣敵逕諒暨麺崧役娶柒訶諦僧爍爱沸薰僖昀塾煜罈刁晒椗夕浪堉耙祈蔚抑秒鱗慾摺饐潑餵甩瑨鄒吔綢糙縱胚涂昂卅擦恤傭甁稀贗熗擱脈蔻盔蝴聶搗薌湛竭衰粵萌敘諸亷斬蛇唷鴆奮桴寸檔鏈碞瀊卿棗惑誘棚諺葫藻屯灃巃瀑巡遮蓆喪䥶惍稚烀蟎譚荏苛濂錵姸淮烙谈蚤末猜昜啉玹鎧虛咩靶徑鯰酢硯琩茨玺猫峇祠壷鷲鈣趴炊泛審萳恕躲籌函緞呢羢沾盎鯊盲坣礫抿汪勃跡萫悸逆屁琉孜絮甭顴嘎吟拒炳押丽醜囋餽筵颺斧鋤薔尗泷衙摯牢鑰弍灑膺稽扣盾醐醍甦侯幹竤贏痰瀉冲崔契衆恬锅坮秉宬銹猿粳璦謬哺泗懿慚愧秦酩醺犁几恪歉咱啥咚刚楀喧嘩啇囯万抙逛賔渦肖碇罕邀麩噘夀晰説災鋭喱鏵袍彤搵歪篤婭剃暫粩埆偏枸籬摘敢琼談鐿轅鉦孚栩肪揉侵氫侖粟葯狄蒟蒻唸慨遵腋抖濺綻途侑厭賒淵隼棨楠貌趾屆擬瑠惶胜萍蝨囊婉咬纜鱉矮閹螃孰巒渠熄湞筴卒紛繽塘埤砥謐泱遼愉漲槤菡旌侶滲塞壅遨壬箏挖税降禱压淅弎剩啞竇暈眩蔭湳丨瓔困矞彗厘悌咒勘稱痕秝蘊焼殯葬羴滕株猪扳炁扎懋毓騼醮壇癸凉妖傍髓踢踐蘑燜苟雖畊奎尭剝槓傲啄遞玬礴徠撿馮濁碾俄絓竂丐壕鱸虞盃魟俐汐药椏氟呆噲塚禽湲釔舵酱鸭传这话：郑总闯涯虾，鱼孙记锡纸饺她们饭题厅馄饨庆连锁贩贵阳农汤圆乐惹馆盖浇兴业书桥烫检兰剁馅啵腾讯应鲜滚给来汉订肘黄焖鸡诚铺绵经济实绿庐忆芦红创种疯价带请楼动嗦风货铭厨卖嘞个尝浆®馋匆储耿强夹馍陕过门奥广场怀旧设菏泽签撩为龙网铁统鱿闻见热卤选还衷乡烧镜优、斓机头华蝎发梦区·领楚车视荐杞缘萝犟轰阁岛纯专长萨饮东挞装哒荣厂怼纤户独宫鹅（）员资议儿岭鹏欢舰铃饼戈牟厠陈苕启择烟现进惊绝浏刘态达鳗—厉蚝韩净钱额杂粮筐运转务｜缤灿腕试减问内询。樐开渔约劳佘轻艺觅钵马宽潼关尔滨让样综肠财贡煨档单县鲁并临簋晗幢购皖块颜¥瘾跷边质赵烩吹闲简撸据鸽贝亲纪吕汇无寻斋护鸣臊丝顺围蛳庙缃▪呀补阴润类较些畅脉帮对杨轩脑项许宾买赠笃尽™识栖凤柠图杀归宁赣粤盘颓医张满槑级馕验昙罗圣ǐà隐状饱费语乌齐闸飞养邻笼银吴炝瑢莲哎麦泾瓯∣歺玛远温溏飘劲楸姝苏与莱错诗脖丛の乖别咥辽贴办属°岁园摆_坚鸳鸯饪么￥弹库谜答节《瀞严憨镇酿抢突观澜谁爷码欧标谭衢际饿荟组说顾锦嗨鲍胗栈葑谊够绪“”编谢屿钟队觉没幺鲩噜牦义驾逍遥鹃妈稣耍岚缆顿圭蚁蚂矿呗钝摄窝荞搅陇狮晖箐针蚬雒←复柜驴证账汆稍戏菋卫匹栋邢莞篷剡帅维笔历适倾潜沧颈训邵诺职诊扬褚糁钢盐绍槿逼兽闺贺甏碚饰帐飮仓随拴桩贱贾层潍叠积纷苹珀琥龄饹饸郞贤嫡监汍喻骄评侠赢馒产导倡丧确蓝岗札树响赖驻辉盗软纱环坛协负啊赛奋叶涛铝鞭俞愚奚浚宠炲横沥称彪诉屠鲅荤载－雾厢苼祛堰爅叼鲈栏…挂报胶幂吻聪糯糍烓鸟亚肤滩荆绚黔纳课贸鄢贯宛Ⅱ凯闽纬瀘琅琊夺扩诱浓郊羔涩捻羯励嵊墙眯预释逐灵勺岂俵将圳伦漓仅谦硕抚郝计杖陆娅翟邮•笨拨烂馈溶销难噢摊弯换搓勾缅啪吓涨妆吗幅术杈泼郫誉膨灯踪细舔愿衹婴邕浔闪襄【丢赔鳍ā濤兑币盱眙搏燃叙腌闫芈谣矾锣跃钥驼禄轮认荀死尧届获崂顶柿臂凭慵懒静淌甚绣渌艳¦揽沂听舆谐疗菟贞胪鉴衔蝉芜稳豌亿缙沤迟忧伟晓枫窑祚鸿烦恼闖潯测荔巫Λ暹兮似离继责啖赞焰乔拱骚扰喷驢仨豉邳终扫恶争‘肃绕贼笋钩诸榧骆访递藩篓莅但覃沟楹洒抛险损蹭萧颗剂抻锋权蒝侬芷庞毕谱祉樱甑→綦躁涧馀潇须纺织军砀麯剑娱链炼锤献謎数则邓赐猩厌沪沌腻熳Áō専卢酆缠汊濮唝坂莴楂蒋伞瑧冈讲详鹵摇偃嵩￣乞谨′剥颐▕？捅缕艰参䬺趁呦饵垦袜熠显缇寇槟絳驿歹耘竞­唛习涡鄂蘸佼ǒ∶裤寳嚸鲢俤∧坟废挪気辘碉雁捣烁Ⓡ脸樊玮★扑渊醴瑶霊呜～备钻冻菽❤咻籣尹岔壮霄浒娇傣熘孤邂逅厕郸莆虽倦怠矣俩录驰续钦党鲫侧叕谛峡⑧澧异朔冯战决棍；﹣丑妇焉坞壳馐帜娄鳯秜结稠齿矫恵缔皲渭鹤蛎鞍赴盼鳌络陂濑馇孃规论凳贰兎阅读◆练墩↓栢枣罉鳅顽倌曰颖巍沽荘啃髙○且阆匣阜遗仑昊构挡巳镖燚栾啫饷韶菒仪脏浅辈靡埠溧睦举钳哟扞侣龚龟钜埭が搽螞蟻娚垵☎骑Ξ谋黍侍赏扮洁闹鶏φ麓喂Ⓑ奖砖仉腱朙莹抺该穇箩嬷廰祗坝蛮槁亖俯浣锨闷梆▫姥烎枪羌穷晕箕婧辅柃「」≡ɔ唔盏睇淖沣礁豇栙涌钓迁妨仍漳鳕拐餸绘漕餠áǎ邨闵锹兹咀篮势卟频甬卞憾駱涉珥熹萦瑭谓兢攸䒩柗鲤粕汾怎ī盯氽镀∙纵逝姣旮旯︳屉㕔а铸韦击伿『』☆声牵夸愁噻嗏嵗湶営汴咑㸆嗖敌炕锌堽邯珑尕喃脱邹烨檐碌页荠囬肫淀鳝鲺龘｝｛犹坏烜媛桓堇↑扛罄鲶鍕携卑遍蓓坨馔姫堵烛阔噤湿鳜崽＋灼腥峭喽郏栀鲨诈斥螄稿掂鹭乱埃禅鹌鹑û囧阡眞嘱鳞浃贷荊讨陌冀砵鵺鹰啜咽鯡蒌蒿③嬢瓮嚣莽咋饥凈隧鉢茫íóΘ祎褥梳湫︵︶伤榄拽舂斩飨执渤惺箬茜俭瀏嫦琶琵咿吖戛舱祁缝羋绒淝锐嗯睁＇既虚娴碱哚逻匪ü槐写踩赚觞鲽闰璐侨煸棵峪π粙喊饶孵屎陦莳倔祸灮莜淩鲮き缪糠埧痴猎嬉凼舅褐醪氷碛绽瀾孖雍òǘ涤辜嘢┐徫ステキ迈虢糐挥纹枊劝俺粢馓拥嘶ㄍ丩ㄝホルモニ颂噫否绎垒锄卧炯轴隴宀荥贪麽暧恨礳叽断岢睾迹禺沭瀛唤愤怒郷狱蠡咶垅绅馏襠郴恺樵呐砸鄞叹遂吆嘛灭蜗泄鑪淄钰壶旱蕴峧赶咏渎靣趟镫讷迭彝却稷庖扉瞿筏韧墟俚翕貂搄寞泸炘茯骏糘毫幽ぉにぃ炽斛窜鲷爲裸窦羡冕粶䬴嚟辆撮隋赁咭崟沩颠诠拙瞄漖橱帼崮勋苍雏睐袭皋陝彻垚咯凑纽巩茏昧坯霾闳凖皱缗箍筹孬唠输驭哼匡偵壩蝇贛漟邴謠怿亁棱阶堔炜笠遏犯罪仝珅咧摑滘颁锈佤佗卌匱藺蔺塍鯽鳟畿耦吨䒕茬枼桼嘍沱楞屹掺挢荻偷辶泮某聂甡吁鎬谅鞘泪鐡犊涪杬睢しんいちょくどぅ戍莼蒡砣惯隣撇筛昵涞绥俑鐉埒侗仗违辫灞琰崖炑昝迩浈挚砚ú缴黒畈忽燍姗逹岫践嗞㥁祯牤诏杠苞滤鞑萤榶轨耒嚮漪键彦词敖鸦秧囚绾镶帷豁煒珲緋仆ラナン瀨缓氵汥殡靳闭偘酔哑佚1234567890abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True,
                        help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    parser.add_argument('--gpu', default='0')

    opt = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    
    if not opt.exp_name:
        opt.exp_name = f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
        opt.exp_name += f'-Seed{opt.manualSeed}'
        # print(opt.exp_name)

    os.makedirs(f'./saved_models/{opt.exp_name}', exist_ok=True)

    """ vocab / character number configuration """
    if opt.sensitive:
        # opt.character += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        opt.character += string.printable[:-6]  # same with ASTER setting (use 94 char).

    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    # print('device count', opt.num_gpu)
    if opt.num_gpu > 1:
        print('------ Use multi-GPU setting ------')
        print('if you stuck too long time with multi-GPU setting, try to set --workers 0')
        # check multi-GPU issue https://github.com/clovaai/deep-text-recognition-benchmark/issues/1
        opt.workers = opt.workers * opt.num_gpu
        opt.batch_size = opt.batch_size * opt.num_gpu

        """ previous version
        print('To equlize batch stats to 1-GPU setting, the batch_size is multiplied with num_gpu and multiplied batch_size is ', opt.batch_size)
        opt.batch_size = opt.batch_size * opt.num_gpu
        print('To equalize the number of epochs to 1-GPU setting, num_iter is divided with num_gpu by default.')
        If you dont care about it, just commnet out these line.)
        opt.num_iter = int(opt.num_iter / opt.num_gpu)
        """

    train(opt)
