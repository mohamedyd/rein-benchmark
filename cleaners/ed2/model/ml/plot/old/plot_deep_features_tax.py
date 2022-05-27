import matplotlib.pyplot as plt
import numpy as np


def plot_list(ranges, list_series, list_names):
    fig = plt.figure()
    ax = plt.subplot(111)

    for i in range(len(list_series)):
        ax.plot(ranges[i], list_series[i], label=list_names[i])

    ax.set_ylabel('fscore')
    ax.set_xlabel('labels')

    ax.legend(loc=4)

    plt.show()

fscore_deepfeatures_avg = []
fscore_deepfeatures_avg.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.094827908600206928, 0.094903930648086415, 0.094903930648086415, 0.094927676228818578, 0.094927676228818578, 0.094963293490047493, 0.09498644564235191, 0.066952802649146492, 0.090752620873102804, 0.090795297062899558, 0.28269844248451864, 0.28273296920550883, 0.28363704401633155, 0.28365998222531086, 0.29686664381213579, 0.38728274912839955, 0.38719761333419805, 0.38719761333419805, 0.46487480863252506, 0.4653110713821752, 0.46531394989174141, 0.51253478297361343, 0.51253478297361343, 0.51232486314662817, 0.51232486314662817, 0.53316041094497002, 0.53689462572828761, 0.53689462572828761, 0.58419966084199659, 0.58419966084199659, 0.58502488878497194, 0.58502488878497194, 0.58502488878497194, 0.59449251980052797, 0.5945096984734225, 0.6110199642376537, 0.6110199642376537, 0.61089934338341978, 0.61089934338341978, 0.61089934338341978, 0.60543537314169993, 0.60549303339156846, 0.61750145063519302, 0.61750145063519302, 0.61969956499772028, 0.61969956499772028, 0.61969956499772028, 0.62459940324897778, 0.62462790540664959, 0.63815709750127236, 0.63815709750127236, 0.64176779568918674, 0.64176779568918674, 0.64176779568918674, 0.64106513951563704, 0.64116271905184752, 0.65241949778434272, 0.65241949778434272, 0.65976900866217514, 0.65976900866217514, 0.65976900866217514, 0.66143458332844896, 0.66143864399627239, 0.67950632903977837, 0.67950632903977837, 0.68864915735722121, 0.68864915735722121, 0.68864915735722121])
fscore_deepfeatures_avg.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.092013669155803071, 0.092041579410220217, 0.092041579410220217, 0.092088697687660148, 0.092088697687660148, 0.094313156918122296, 0.094349116892265897, 0.013334151021087582, 0.013320830448614018, 0.013856328073060637, 0.24418838450235869, 0.24422722984363646, 0.24795027975387857, 0.24796423562955677, 0.25605556963346104, 0.37513454207087155, 0.37561490786804147, 0.37561490786804147, 0.45832344174513989, 0.46038235255578708, 0.46045125585355462, 0.47970645598492623, 0.47970645598492623, 0.47967834517054847, 0.47967834517054847, 0.50229964430430873, 0.49955978396562373, 0.49945141941121196, 0.54884203025297185, 0.54884203025297185, 0.54925334456285835, 0.54925334456285835, 0.54925334456285835, 0.55248047178107706, 0.55269796504482593, 0.6129046284965306, 0.6129046284965306, 0.6137146047204044, 0.6137146047204044, 0.6137146047204044, 0.6133027238319787, 0.61354477935797447, 0.64911121822910156, 0.64911121822910156, 0.65223189677144022, 0.65223189677144022, 0.65223189677144022, 0.65659423414271123, 0.65659900425818418, 0.66432564475971256, 0.66432564475971256, 0.66563263771385595, 0.66563263771385595, 0.66563263771385595, 0.67383700594613494, 0.67388845485605009, 0.68274935822240357, 0.68274935822240357, 0.68737055367754529, 0.68737055367754529, 0.68737055367754529, 0.68947466394399737, 0.68947068417696167, 0.68263284143880509, 0.68263284143880509, 0.68575212065625024, 0.68575212065625024, 0.68575212065625024])
fscore_deepfeatures_avg.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.027265719265317064, 0.02731416314766405, 0.02731416314766405, 0.027527854569824917, 0.027527854569824917, 0.027577425380948793, 0.027650541371138593, 0.021160172867279538, 0.021177803902331756, 0.021228010825439785, 0.24383360105688251, 0.24383360105688251, 0.24382041870275706, 0.24388229375502143, 0.24640797501728579, 0.36658315285857856, 0.36647638963461149, 0.36647638963461149, 0.39374509381664596, 0.39383456975354952, 0.39384299070613887, 0.39244959244959249, 0.39416037540443827, 0.39489115821390769, 0.39489115821390769, 0.47768259846311401, 0.47839085602717163, 0.47846598072308766, 0.4777083489151645, 0.4777083489151645, 0.48052256370949697, 0.48052256370949697, 0.48052256370949697, 0.48555765109880822, 0.48552615567664037, 0.48735476898135632, 0.48735476898135632, 0.49635104978521788, 0.49635104978521788, 0.49635104978521788, 0.50013034672228129, 0.50020053475935822, 0.51032666174445751, 0.51032666174445751, 0.52636410128640554, 0.52636410128640554, 0.52636410128640554, 0.53400116087208882, 0.53402199006143136, 0.59641152631154304, 0.59641152631154304, 0.59568697945499438, 0.59568697945499438, 0.59568697945499438, 0.59944672563297807, 0.5994926596912542, 0.64575096460725001, 0.64575096460725001, 0.64518092782582226, 0.64518092782582226, 0.64518092782582226, 0.64518092782582226, 0.64520441659206207, 0.65997263146670448, 0.65997263146670448, 0.66192737677354652, 0.66192737677354652, 0.66192737677354652])
fscore_deepfeatures_avg.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.042377832654301277, 0.042453748782862705, 0.042453748782862705, 0.042875491378974812, 0.042875491378974812, 0.042921194383037015, 0.042921194383037015, 0.0070303094214934133, 0.16551936128390346, 0.16551936128390346, 0.36384110645117046, 0.36384110645117046, 0.36063225522801889, 0.36065550520844414, 0.36161656780938978, 0.36161656780938978, 0.36245796918576723, 0.36245796918576723, 0.36248144935225085, 0.36609279252990951, 0.36613903275220655, 0.36618419704664201, 0.36618419704664201, 0.36866991623128337, 0.36866991623128337, 0.37188945126169759, 0.37315137451170094, 0.37326488616780701, 0.37335486828033265, 0.37335486828033265, 0.396321094209236, 0.396321094209236, 0.50093477984091983, 0.50708999793138887, 0.50699461338198382, 0.53075935099997384, 0.53075935099997384, 0.54150331039854604, 0.54150331039854604, 0.54150331039854604, 0.54399067266897694, 0.54423267599427128, 0.64471156155434939, 0.64471156155434939, 0.64433380235956272, 0.64433380235956272, 0.64433380235956272, 0.65097325857622412, 0.65111149735985652, 0.67684718723761528, 0.67684718723761528, 0.68177275790745784, 0.68177275790745784, 0.68177275790745784, 0.68510265934944725, 0.68526395086052028, 0.70595990836826283, 0.70595990836826283, 0.70581769820072326, 0.70581769820072326, 0.70581769820072326, 0.70875105457493781, 0.70876035955105421, 0.72858277733643684, 0.72858277733643684, 0.7281004552503878, 0.7281004552503878, 0.7281004552503878])
fscore_deepfeatures_avg.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.085932104885303062, 0.086011361538934808, 0.086011361538934808, 0.086152920750796322, 0.086152920750796322, 0.086335458913425325, 0.088209453523106662, 0.034506646237601968, 0.036050998974158567, 0.036098966856361284, 0.19569298895197063, 0.19571998115056358, 0.19956266372123918, 0.19900962487846427, 0.19326427785054762, 0.32443112343801256, 0.32522771451340221, 0.3751463213723138, 0.40973381233911343, 0.41107020977439884, 0.41155607784282772, 0.47351289771340221, 0.47351289771340221, 0.47357422961585943, 0.47357422961585943, 0.54274240559621734, 0.54274240559621734, 0.54278772958439969, 0.54370008149959259, 0.54370008149959259, 0.54547692626641531, 0.54547692626641531, 0.54547692626641531, 0.54547692626641531, 0.54554406058289662, 0.54810137229148737, 0.54810137229148737, 0.55507379895502484, 0.55507379895502484, 0.55507379895502484, 0.55507379895502484, 0.55551408124388613, 0.5703907261758111, 0.5703907261758111, 0.57723260317380487, 0.57723260317380487, 0.57723260317380487, 0.57723260317380487, 0.57718416090509117, 0.59608420947481455, 0.59608420947481455, 0.60764886555038145, 0.60764886555038145, 0.60764886555038145, 0.60764886555038145, 0.60759085725618522, 0.65678997577678755, 0.65678997577678755, 0.66193706852353063, 0.66193706852353063, 0.66193706852353063, 0.66193706852353063, 0.66201749994039527, 0.69846841097638801, 0.69846841097638801, 0.70092815205097159, 0.70092815205097159, 0.70092815205097159])

average_deepfeatures = list(np.mean(np.matrix(fscore_deepfeatures_avg), axis=0).A1)


fscore_deepfeatures_last = []
fscore_deepfeatures_last.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.052206652007043108, 0.052268172561237822, 0.052268172561237822, 0.052314307735994485, 0.052314307735994485, 0.052314307735994485, 0.052314307735994485, 0.077721889861515028, 0.12301816593147963, 0.12301816593147963, 0.24091611522764197, 0.24091611522764197, 0.24092773224675249, 0.24118753092528455, 0.23056201505767507, 0.31449584746754006, 0.31452049186320236, 0.37951248313090419, 0.37954665260938319, 0.379694703699539, 0.38151841825052718, 0.43846803137502494, 0.43846803137502494, 0.43847742696960751, 0.43847742696960751, 0.44023277652493986, 0.44445811461067369, 0.44341262832542222, 0.47791802352181284, 0.47791802352181284, 0.47924284792428479, 0.47924284792428479, 0.57359922829095455, 0.57345532274438349, 0.57357061719045821, 0.6626047129737298, 0.6626047129737298, 0.66704118078387775, 0.66704118078387775, 0.66704118078387775, 0.66937802149801362, 0.66957284996435995, 0.66234333758674335, 0.66234333758674335, 0.67156003632430428, 0.67156003632430428, 0.67156003632430428, 0.67414005091291129, 0.67404675468818653, 0.69819010017746386, 0.69819010017746386, 0.70599522036858309, 0.70599522036858309, 0.70599522036858309, 0.70669292468809042, 0.70669071939883488, 0.71696772825968635, 0.71696772825968635, 0.71718670340434554, 0.71718670340434554, 0.71718670340434554, 0.71756048675733708, 0.71756879000583973, 0.72382830983586921, 0.72382830983586921, 0.72385741079623511, 0.72385741079623511, 0.72385741079623511])
fscore_deepfeatures_last.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.088205035878706831, 0.088237941543328147, 0.088237941543328147, 0.088786338190208702, 0.088786338190208702, 0.088810334854425091, 0.088894601542416449, 0.032566558938032619, 0.033800862368042819, 0.033833344346791777, 0.25002389512612955, 0.24910191857880892, 0.2493280989387969, 0.25032030169886577, 0.24511046516040405, 0.36612771009412892, 0.36696472576216294, 0.36696472576216294, 0.36946162137629773, 0.36978897334362842, 0.36916210011585177, 0.3726285997514564, 0.3726285997514564, 0.37715261975349412, 0.37715261975349412, 0.48554355043158087, 0.4869098712446352, 0.48709655774005234, 0.53173780706385365, 0.53173780706385365, 0.54251430490435415, 0.54251430490435415, 0.54251430490435415, 0.54349201430581917, 0.54398975824545293, 0.58689121357358343, 0.58689121357358343, 0.58181037693565396, 0.58181037693565396, 0.58181037693565396, 0.58159664130812205, 0.58176950471042865, 0.62726606016158648, 0.62726606016158648, 0.63377357922921873, 0.63377357922921873, 0.63377357922921873, 0.63621739078238104, 0.63610971253206749, 0.65632379721553191, 0.65632379721553191, 0.65485125024958002, 0.65485125024958002, 0.65485125024958002, 0.66414737836561166, 0.66415748507870775, 0.68278528869302413, 0.68278528869302413, 0.68804798467793271, 0.68804798467793271, 0.68804798467793271, 0.6921355243174111, 0.69214349888696236, 0.69751109960666036, 0.69751109960666036, 0.70299874002519946, 0.70299874002519946, 0.70299874002519946])
fscore_deepfeatures_last.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06957575233109059, 0.069617514710972661, 0.069617514710972661, 0.071251548946716231, 0.071251548946716231, 0.076910369396740577, 0.078217624259005927, 0.026363730106898568, 0.051064479724397753, 0.053602069086739264, 0.16322242234650991, 0.16325144166205863, 0.1609735269000854, 0.16033506182688473, 0.1577784277404049, 0.2749281029839164, 0.27563579229501373, 0.37280054654737593, 0.37280054654737593, 0.38492667479387432, 0.38518550561635739, 0.43207357298596749, 0.43207357298596749, 0.43505255098489598, 0.43505255098489598, 0.53694786433795783, 0.53889585539695017, 0.53900902656016625, 0.59236162315507956, 0.59236162315507956, 0.59774368174824177, 0.59774368174824177, 0.59774368174824177, 0.59711001396369434, 0.59739595993592742, 0.63238313115198208, 0.63238313115198208, 0.64210790404281748, 0.64210790404281748, 0.64210790404281748, 0.64328218350671751, 0.64326627049473761, 0.64512663176806961, 0.64512663176806961, 0.65548041417979008, 0.65548041417979008, 0.65548041417979008, 0.65637884460452123, 0.65668441596133453, 0.67300157308458086, 0.67300157308458086, 0.67389198001821804, 0.67389198001821804, 0.67389198001821804, 0.67449279060865119, 0.67456252172905318, 0.70219914820932927, 0.70219914820932927, 0.70247831753022061, 0.70247831753022061, 0.70247831753022061, 0.70336024070685177, 0.70369164439853593, 0.70756479037156894, 0.70756479037156894, 0.70757753819426705, 0.70757753819426705, 0.70757753819426705])
fscore_deepfeatures_last.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.057602235906571042, 0.057667505987840074, 0.057667505987840074, 0.057786784573814788, 0.057786784573814788, 0.057845974346968379, 0.057890252607824627, 0.061820948342921622, 0.10671900446820061, 0.10685684222832759, 0.31749801470970235, 0.31749801470970235, 0.31765020185229165, 0.31762802914508736, 0.41765189177145573, 0.4917706397663924, 0.49075458336611849, 0.49075458336611849, 0.49336265807870555, 0.49346790239815691, 0.49351601521330707, 0.51298887786538561, 0.51298887786538561, 0.51417325417391024, 0.51417325417391024, 0.60344434236245137, 0.60336194563662371, 0.60337896164755211, 0.60204724214997463, 0.60204724214997463, 0.60352988616272218, 0.60352988616272218, 0.60352988616272218, 0.60430655531375355, 0.6043079822918862, 0.63584346704254957, 0.63584346704254957, 0.63687157507670644, 0.63687157507670644, 0.63687157507670644, 0.63835673612974386, 0.6383217116633888, 0.65132278803015586, 0.65132278803015586, 0.65304970421699582, 0.65304970421699582, 0.65304970421699582, 0.65523209346987354, 0.65532017692701483, 0.66128128879521897, 0.66128128879521897, 0.66237953269084104, 0.66237953269084104, 0.66237953269084104, 0.66672805649398859, 0.66678560534993858, 0.6567314539801643, 0.6567314539801643, 0.66395020718271736, 0.66395020718271736, 0.66395020718271736, 0.6711809441158334, 0.67123991279633832, 0.69215614404838266, 0.69215614404838266, 0.7092041585019454, 0.7092041585019454, 0.7092041585019454])
fscore_deepfeatures_last.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.07140668523676881, 0.071444487772268958, 0.071444487772268958, 0.072303077126530391, 0.072303077126530391, 0.072343328043653787, 0.076091091907168867, 0.010386064355580133, 0.01038651308683367, 0.010420897095801398, 0.23784669074879083, 0.23784669074879083, 0.23791997918754926, 0.23818359067895217, 0.23789721211738618, 0.36477738722905684, 0.36483973467266045, 0.36483973467266045, 0.36486367762907429, 0.36528015695231403, 0.36531224833443154, 0.36600388150426599, 0.36600388150426599, 0.36599900436323168, 0.36599900436323168, 0.42181478784107707, 0.42200745176502202, 0.42203352682074957, 0.43437471472610195, 0.43437471472610195, 0.43440077564039514, 0.43440077564039514, 0.48932607501915143, 0.49116846906595568, 0.4911941217464259, 0.54765016875085348, 0.54765016875085348, 0.54765961320865142, 0.54765961320865142, 0.54765961320865142, 0.54777820198399085, 0.54779378454290117, 0.60690447228516919, 0.60690447228516919, 0.60731723878247668, 0.60731723878247668, 0.60731723878247668, 0.61103884494695115, 0.61108234075608492, 0.61979875065390655, 0.61979875065390655, 0.62184347313188915, 0.62184347313188915, 0.62184347313188915, 0.62282735357252372, 0.62283197998184636, 0.64346632801073755, 0.64346632801073755, 0.65341163678509773, 0.65341163678509773, 0.65341163678509773, 0.66060065417781744, 0.6606165849945288, 0.67652149989109633, 0.67652149989109633, 0.68339462356395564, 0.68339462356395564, 0.68339462356395564])

last_deepfeatures = list(np.mean(np.matrix(fscore_deepfeatures_last), axis=0).A1)



fscore_0 = []
fscore_0.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.067932792365537606, 0.067973837251461139, 0.067973837251461139, 0.068029874555869771, 0.068029874555869771, 0.071056289089645591, 0.071104420141427149, 0.12739216987392168, 0.2368906077209994, 0.2369016231616039, 0.38793743644818174, 0.38793743644818174, 0.39790421872170922, 0.39798123818899866, 0.38491582877659686, 0.39038361271614369, 0.39052747346921285, 0.39052747346921285, 0.39055046133630417, 0.37894888824718448, 0.37898946730098687, 0.43675274487646309, 0.43675274487646309, 0.44007670182166825, 0.44007670182166825, 0.44027813474669875, 0.4408953233111958, 0.44123485237573051, 0.49084690401129838, 0.49084690401129838, 0.488833380336543, 0.488833380336543, 0.58381317374313269, 0.58823160364633331, 0.5888044316052663, 0.63207087917437443, 0.63207087917437443, 0.63781969735518973, 0.63781969735518973, 0.63781969735518973, 0.63938136095748677, 0.63970166726003341, 0.65984536111629599, 0.65984536111629599, 0.67054693895860662, 0.67054693895860662, 0.67054693895860662, 0.67226608432723001, 0.67194012936813829, 0.69994170295943214, 0.69994170295943214, 0.70627103994177043, 0.70627103994177043, 0.70627103994177043, 0.71064263855831356, 0.71118163785775002, 0.72252449535360341, 0.72252449535360341, 0.72198223805056694, 0.72198223805056694, 0.72198223805056694, 0.72622220231563384, 0.72649830971410667, 0.73167188198480104, 0.73167188198480104, 0.73268222404637051, 0.73268222404637051, 0.73268222404637051])
fscore_0.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03323242958673335, 0.033280415881815864, 0.033280415881815864, 0.03402428896800664, 0.03402428896800664, 0.038267230915592289, 0.040867133319322978, 0.087295337355255326, 0.12600052332485726, 0.12636740349895403, 0.29611768127181076, 0.29611768127181076, 0.30231468021099722, 0.30149709387659251, 0.30719053098799937, 0.37981773783450395, 0.38115384858983214, 0.38115384858983214, 0.3811743544597222, 0.38955122364385353, 0.39366385786138869, 0.42566404491406479, 0.42566404491406479, 0.42577338701133599, 0.42577338701133599, 0.45199307210480399, 0.45725109629206778, 0.45852126858709152, 0.53454588997498009, 0.53454588997498009, 0.53488254864401652, 0.53488254864401652, 0.60055852472434879, 0.60207263224294849, 0.60266074056277974, 0.6371655133497669, 0.6371655133497669, 0.63729618959935852, 0.63729618959935852, 0.63729618959935852, 0.63753109341712189, 0.63750119058957999, 0.65620278073864091, 0.65620278073864091, 0.65617129938737984, 0.65617129938737984, 0.65617129938737984, 0.65679680136749741, 0.65748636690416218, 0.66658471768576077, 0.66658471768576077, 0.66871961480927467, 0.66871961480927467, 0.66871961480927467, 0.66871961480927467, 0.66847489331852461, 0.69342606467553569, 0.69342606467553569, 0.69526681286549719, 0.69526681286549719, 0.69526681286549719, 0.69526681286549719, 0.69626272766384634, 0.69687711762206117, 0.69687711762206117, 0.69951190448910372, 0.69951190448910372, 0.69951190448910372])
fscore_0.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.062822092964588697, 0.062862167382592912, 0.062862167382592912, 0.063029691553761882, 0.063029691553761882, 0.065419028267921425, 0.066906997580229832, 0.018495516496231136, 0.05927938137457503, 0.062426442554743312, 0.27652202120287228, 0.27652202120287228, 0.28007106591437891, 0.28094726320298818, 0.32194196797390789, 0.4062444562710662, 0.40599624381082466, 0.40599624381082466, 0.40606155800342819, 0.41090733656322403, 0.41112039168551073, 0.48181205579077185, 0.48181205579077185, 0.48271782411235642, 0.48271782411235642, 0.57347383025547805, 0.57390398409714161, 0.57389503175866996, 0.60559344904711732, 0.60559344904711732, 0.60593719726279283, 0.60593719726279283, 0.60593719726279283, 0.61112135856666783, 0.61120062977404399, 0.64248802472922917, 0.64248802472922917, 0.64500253140710917, 0.64500253140710917, 0.64500253140710917, 0.64855276759703029, 0.64847687409687382, 0.67320201974252825, 0.67320201974252825, 0.67652044744555762, 0.67652044744555762, 0.67652044744555762, 0.67949225599891383, 0.67957704645360562, 0.70307862979293712, 0.70307862979293712, 0.70621545099096672, 0.70621545099096672, 0.70621545099096672, 0.70622998707937767, 0.70627555835168299, 0.70309649839411803, 0.70309649839411803, 0.71198151562951284, 0.71198151562951284, 0.71198151562951284, 0.71198151562951284, 0.71215107917663845, 0.71301213854393442, 0.71301213854393442, 0.71693986632000639, 0.71693986632000639, 0.71693986632000639])
fscore_0.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.078102094330922175, 0.078141557601530354, 0.078141557601530354, 0.083267019729440506, 0.083267019729440506, 0.088427725776519514, 0.088465966813410082, 0.10220324250784173, 0.10220589347001105, 0.10351770161746408, 0.29131492734119052, 0.29131492734119052, 0.29141278794815734, 0.29148917257553897, 0.27260936725831364, 0.3881039476726002, 0.38804344125557483, 0.38851526125193997, 0.38869075168114126, 0.39256861518374447, 0.39277560061339239, 0.47476627894336343, 0.47476627894336343, 0.47956705564011504, 0.47956705564011504, 0.57583150104234659, 0.58183172284951501, 0.58182070636944283, 0.61330269061036879, 0.61330269061036879, 0.61543161226073295, 0.61543161226073295, 0.61543161226073295, 0.62060443055227443, 0.62066542699140503, 0.62489403355553252, 0.62489403355553252, 0.62422542083434984, 0.62422542083434984, 0.62422542083434984, 0.62930897634718475, 0.62931460845672427, 0.63121718727984466, 0.63121718727984466, 0.63594937327436651, 0.63594937327436651, 0.63594937327436651, 0.64234533220888124, 0.64246474005263221, 0.64559854802433625, 0.64559854802433625, 0.6498224019905402, 0.6498224019905402, 0.6498224019905402, 0.6530479935168233, 0.65291477668178644, 0.68939627627068334, 0.68939627627068334, 0.69560230774577714, 0.69560230774577714, 0.69560230774577714, 0.69294147724631627, 0.69304156226200109, 0.72042253521126753, 0.72042253521126753, 0.72422263224024541, 0.72422263224024541, 0.72422263224024541])
fscore_0.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.050939388004158315, 0.050982666797236925, 0.050982666797236925, 0.051070753331274428, 0.051070753331274428, 0.051100113739934769, 0.051156905278380328, 0.071833394314558582, 0.077599286033020978, 0.093788684191363866, 0.28142643312938215, 0.28142643312938215, 0.28657234819056071, 0.28657609014996699, 0.27599636505257691, 0.38608681545969475, 0.38613502554038254, 0.38613502554038254, 0.38615445631214013, 0.38936467111743456, 0.389694129861614, 0.39288260286863042, 0.39288260286863042, 0.39972271933113113, 0.39972271933113113, 0.49276810638312057, 0.49981443189650604, 0.49973141276883593, 0.57456689824481477, 0.57456689824481477, 0.57922010541168134, 0.57922010541168134, 0.5906255099607095, 0.59265759469471413, 0.59265942383118431, 0.64690312334568545, 0.64690312334568545, 0.64643218257980639, 0.64643218257980639, 0.64643218257980639, 0.64645102242346875, 0.64663846811997627, 0.68618954509852537, 0.68618954509852537, 0.6918044927178475, 0.6918044927178475, 0.6918044927178475, 0.69196939439495586, 0.69197720593184198, 0.67716954461043821, 0.67716954461043821, 0.68061130831923378, 0.68061130831923378, 0.68061130831923378, 0.68044133506622795, 0.68046185494038802, 0.69592777670412198, 0.69592777670412198, 0.69551432532602742, 0.69551432532602742, 0.69551432532602742, 0.69783161225941259, 0.69796037649497955, 0.71601283557805295, 0.71601283557805295, 0.71724246922499491, 0.71724246922499491, 0.71724246922499491])

average_full = list(np.mean(np.matrix(fscore_0), axis=0).A1)
label_full = [4, 8, 12, 16, 20, 24, 28, 38, 48, 58, 68, 78, 88, 98, 108, 118, 128, 138, 148, 158, 168, 178, 188, 198, 208, 218, 228, 238, 248, 258, 268, 278, 288, 298, 308, 318, 328, 338, 348, 358, 368, 378, 388, 398, 408, 418, 428, 438, 448, 458, 468, 478, 488, 498, 508, 518, 528, 538, 548, 558, 568, 578, 588, 598, 608, 618, 628, 638, 648, 658, 668, 678, 688, 698, 708, 718, 728]
ranges = [label_full, label_full, label_full]
list = [average_deepfeatures, average_full, last_deepfeatures]
names = ["LSTM full row - avg", "bag-of-characters", "LSTM full row - last"]

plot_list(ranges, list, names)