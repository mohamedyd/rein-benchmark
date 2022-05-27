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




fscore_metadata_no_svd = []
fscore_metadata_no_svd.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.12597449040410061, 0.56523703267669645, 0.64925734616456265, 0.77346453413465477, 0.77347408246435545, 0.77347408246435545, 0.77347408246435545, 0.88271919397747201, 0.86138539796529023, 0.95232481949769732, 0.95232481949769732, 0.95232481949769732, 0.95235109521316452, 0.95236154564777131, 0.95250440180905716, 0.98040121081524811, 0.98040121081524811, 0.98040121081524811, 0.98042082716073142, 0.9804404427514205, 0.98046634516724818, 0.98005244457853369, 0.98526447509731774, 0.96528945246829878, 0.96528945246829878, 0.96425617190499535, 0.96436351833009781, 0.96498697733901018, 0.96601540701390121, 0.9664572190846642, 0.98945138255170617, 0.98945138255170617])
fscore_metadata_no_svd.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.26926963207029103, 0.65016596355226242, 0.74936013682159619, 0.85655096850499401, 0.85655096850499401, 0.85655096850499401, 0.856224841341795, 0.90246310919121098, 0.89740905611938449, 0.95954423705373471, 0.95954423705373471, 0.95953233041309749, 0.95953570017424206, 0.95988025310139924, 0.95824418747454909, 0.9691781820538159, 0.97941233078847767, 0.97941233078847767, 0.9794248979906075, 0.9794248979906075, 0.97945108807226455, 0.97794373718221561, 0.98097199633989796, 0.98097199633989796, 0.98097199633989796, 0.98097833994548611, 0.98107388105060334, 0.9815611912517449, 0.98499375008054013, 0.98691241035677035, 0.98691241035677035, 0.98691241035677035])
fscore_metadata_no_svd.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.11108443857648682, 0.57333877255008614, 0.63468442686924398, 0.76153168460134679, 0.76153168460134679, 0.76153168460134679, 0.76153168460134679, 0.87922584784274582, 0.85042814561514979, 0.94669871887701407, 0.9467109052935101, 0.94671768450252791, 0.94672785315243413, 0.94673124265877173, 0.94410962420974942, 0.9705184881576252, 0.98054677760705067, 0.98054677760705067, 0.98054677760705067, 0.98055687982373629, 0.9805768326386517, 0.97223854864621295, 0.97139830508474578, 0.9716370245189212, 0.9716370245189212, 0.9716370245189212, 0.97175765380328727, 0.97189169757675786, 0.98341249071112413, 0.98620256224317138, 0.99109478720031763, 0.99109478720031763])
fscore_metadata_no_svd.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.21471278973462971, 0.62444763998583863, 0.6892881956723429, 0.80658370740761787, 0.80658370740761787, 0.80654654813894555, 0.80655106848775471, 0.87881001157922856, 0.87445608494457294, 0.97509149470395728, 0.97509149470395728, 0.97508899094115786, 0.97514467471186095, 0.97506978845626324, 0.9577462120614153, 0.96365195054781139, 0.96375607964054366, 0.96375607964054366, 0.96377121038947988, 0.96376818420168486, 0.96384036376311444, 0.97274691750349562, 0.97605600038356433, 0.97605600038356433, 0.97605600038356433, 0.97606254614728138, 0.97606893884426438, 0.97607517859711368, 0.97607845142813299, 0.97818059965852344, 0.97818059965852344, 0.97818059965852344])
fscore_metadata_no_svd.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.13060744052219198, 0.56787885688261419, 0.62258118205629087, 0.7517405075851995, 0.7517504167658966, 0.7517504167658966, 0.7517504167658966, 0.86383170982957724, 0.87102995736794675, 0.97151907018140893, 0.97151907018140893, 0.97151626403327085, 0.97150074274757703, 0.97150111574277054, 0.96934942442155614, 0.96481549669919253, 0.96840921648265366, 0.96840921648265366, 0.96840255002633402, 0.96842152880772125, 0.96843112038826162, 0.97047776563401711, 0.97745294827591733, 0.98428789300419472, 0.98428789300419472, 0.98429535437818172, 0.98430170137444739, 0.98430825082295426, 0.98464238135021886, 0.986014932879415, 0.986014932879415, 0.986014932879415])

average_metadata_no_svd = list(np.mean(np.matrix(fscore_metadata_no_svd), axis=0).A1)



fscore_metadata_no_svd_mlp = []
fscore_metadata_no_svd_mlp.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.23961980711130468, 0.63212250866786135, 0.72285828841270561, 0.79784921148359866, 0.79784921148359866, 0.79784921148359866, 0.79785771100309322, 0.85713419175257044, 0.85946696335835271, 0.93010800171591512, 0.96789629143419753, 0.96736991261906635, 0.96738007380073798, 0.96733651036251589, 0.95840066455100614, 0.96344095721032719, 0.97341569405713779, 0.97341569405713779, 0.97394085855022683, 0.97394085855022683, 0.97398716598027013, 0.97035511571293287, 0.97219405352652644, 0.97219405352652644, 0.97219405352652644, 0.97219405352652644, 0.97219405352652644, 0.97219714405605073, 0.97621831611365628, 0.97672947939955279, 0.97672947939955279, 0.97672947939955279])
fscore_metadata_no_svd_mlp.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.14043546338148835, 0.57365964196691588, 0.61723636906355761, 0.74715729438856471, 0.74715729438856471, 0.74715729438856471, 0.74715729438856471, 0.85654261486513505, 0.85781087194083716, 0.96672605702789605, 0.96672605702789605, 0.9667328503241247, 0.96676286328466532, 0.96676965553510374, 0.95632508099080105, 0.9613250389204343, 0.97160564497302548, 0.97160564497302548, 0.97160893368808843, 0.97159012770530229, 0.97161315082512478, 0.98133304051828263, 0.98187267272310375, 0.98665459867508853, 0.98665459867508853, 0.98665459867508853, 0.98665785106990744, 0.98742808028068729, 0.98840506182789656, 0.99214988218420241, 0.99223748679027768, 0.99223748679027768])
fscore_metadata_no_svd_mlp.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.11453063181872676, 0.57720794118178365, 0.65611110165017628, 0.77892366967366589, 0.77892366967366589, 0.77892366967366589, 0.77892366967366589, 0.89118956631490376, 0.88085153062488286, 0.98049864390272212, 0.98049864390272212, 0.98050521957024406, 0.98045513322820832, 0.98046828393704188, 0.97989549655254671, 0.98077933393985217, 0.97987705496592692, 0.97987705496592692, 0.97987705496592692, 0.9799399353149546, 0.9799399353149546, 0.98120561901821546, 0.98547751201095946, 0.98332254404912223, 0.98332254404912223, 0.98334209278178719, 0.98334209278178719, 0.98334860885895037, 0.98317402435937284, 0.98802933341008892, 0.99063777923889851, 0.99063777923889851])
fscore_metadata_no_svd_mlp.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.26377874702616977, 0.66526980579400974, 0.79121367783485386, 0.89159320032058043, 0.89159320032058043, 0.89159320032058043, 0.89159320032058043, 0.93737472070957517, 0.92334684780804699, 0.97341394411502546, 0.97341394411502546, 0.97341760296601554, 0.97342091835086964, 0.97336782243923237, 0.96493774739575011, 0.97439872393989968, 0.97439872393989968, 0.97439872393989968, 0.97437687769443149, 0.97439658706569643, 0.97445292881668633, 0.97445621362612866, 0.97792643848383098, 0.97792643848383098, 0.97792643848383098, 0.97794856900288207, 0.97794856900288207, 0.97795481167274678, 0.97680745729450069, 0.97704675707791366, 0.97704675707791366, 0.97704675707791366])
fscore_metadata_no_svd_mlp.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.12495757943808386, 0.56405650139976837, 0.61313065036748027, 0.74381259697420543, 0.74381259697420543, 0.74381259697420543, 0.74381259697420543, 0.85988196844995624, 0.87238561270834381, 0.98607069814254478, 0.98607069814254478, 0.9860708778766315, 0.98610685496227524, 0.9861178307175249, 0.98584526869479117, 0.98178394583704198, 0.98178394583704198, 0.98178394583704198, 0.98179025536057996, 0.98181970078659186, 0.98188688703162796, 0.98227314767238727, 0.985633580660218, 0.98994194442932304, 0.98994194442932304, 0.98973910760650674, 0.98975204494885627, 0.99071494596245679, 0.9907958298299514, 0.99215016409803658, 0.99142908236255867, 0.99142908236255867])

average_metadata_no_svd_mlp = list(np.mean(np.matrix(fscore_metadata_no_svd_mlp), axis=0).A1)


fscore_metadata_no_svd_mlp1 = []
fscore_metadata_no_svd_mlp1.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.049231227799995031, 0.52095839395095045, 0.72943285896586807, 0.83999056339481881, 0.83999056339481881, 0.83999483937556452, 0.84000884662169628, 0.96151296368979489, 0.96438678547598788, 0.97867367047265952, 0.97877535382527414, 0.97878861669832318, 0.97878861669832318, 0.9787153914082386, 0.97608435488566159, 0.97862476634458007, 0.97862476634458007, 0.97862476634458007, 0.97864125311535533, 0.97865114492234295, 0.97872711364803033, 0.98129648345383402, 0.98275070335995862, 0.9708704126665213, 0.9708704126665213, 0.97064424744104372, 0.97065400017685921, 0.9706572510477296, 0.97102025866424169, 0.97373394992502615, 0.98668914948519459, 0.98668914948519459])
fscore_metadata_no_svd_mlp1.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.12086520076481834, 0.56214481330267063, 0.61996472786937029, 0.74956075145289913, 0.74956075145289913, 0.74956075145289913, 0.74956075145289913, 0.86561908026484335, 0.86966901322365153, 0.96678387217704942, 0.96678387217704942, 0.96677492849871538, 0.96678203599393753, 0.96679873063474731, 0.96532569459901474, 0.96527585116747217, 0.97784013401550762, 0.97784013401550762, 0.97785597635897037, 0.97766273483285115, 0.97776208508391438, 0.97959806004768779, 0.9861343887785885, 0.99079023264756771, 0.99079023264756771, 0.99079023264756771, 0.99100005769193789, 0.99192734934627869, 0.99210960131880099, 0.99048577674913607, 0.99048895186133801, 0.99048895186133801])
fscore_metadata_no_svd_mlp1.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3402477883526896, 0.70768216716779853, 0.81983112649175316, 0.91500590568652762, 0.91500590568652762, 0.91500590568652762, 0.91500590568652762, 0.93091446407450473, 0.92067029146982804, 0.96532105566060999, 0.96532105566060999, 0.96532786103519841, 0.96533806892922902, 0.96533806892922902, 0.96483504217102822, 0.96834522952000524, 0.97869471500575345, 0.97869471500575345, 0.97869471500575345, 0.97870147381522687, 0.97869899199429544, 0.97938903799284094, 0.98203775050992326, 0.98203775050992326, 0.98203775050992326, 0.9820410462620317, 0.98204151140090457, 0.9820543452571735, 0.98219771895122254, 0.98704688255195738, 0.99097932360955276, 0.99097932360955276])
fscore_metadata_no_svd_mlp1.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1651003035617524, 0.57857474517434282, 0.69085698701066101, 0.80570543695016827, 0.80570543695016827, 0.80570543695016827, 0.80572761166317586, 0.89540333514566317, 0.91530369399510225, 0.98415037749508749, 0.98415037749508749, 0.98416022701196781, 0.98401116762423013, 0.98351818616121578, 0.96894854469685376, 0.96154832011161406, 0.96157190505236667, 0.96157190505236667, 0.96157190505236667, 0.9617211838006231, 0.96222210672660891, 0.97667118912804585, 0.98419145334354619, 0.98419145334354619, 0.98419145334354619, 0.98419472256283014, 0.98419799176107103, 0.98487367930141678, 0.98558738711513061, 0.98721621082309319, 0.98721621082309319, 0.98721621082309319])
fscore_metadata_no_svd_mlp1.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.12949888270812535, 0.56894733013240206, 0.68095182217453976, 0.79977882855399629, 0.79977882855399629, 0.79977882855399629, 0.79977882855399629, 0.90448390638342369, 0.90834509374230665, 0.96324373173482958, 0.96324373173482958, 0.96319748278843587, 0.96323455321417306, 0.96323213937463437, 0.96316569586020384, 0.96658576471809776, 0.98355970397767878, 0.98355970397767878, 0.98361715817347639, 0.98389969389701115, 0.98390595049924201, 0.9807760963607417, 0.98121226539443318, 0.98121226539443318, 0.98121226539443318, 0.98077497758740917, 0.98076187721919628, 0.98076502872713711, 0.98351760157362822, 0.98589107315501134, 0.98589107315501134, 0.98589107315501134])

average_metadata_no_svd_mlp1 = list(np.mean(np.matrix(fscore_metadata_no_svd_mlp1), axis=0).A1)






label_0 = [4, 8, 12, 16, 20, 24, 28, 38, 48, 58, 68, 78, 88, 98, 108, 118, 128, 138, 148, 158, 168, 178, 188, 198, 208, 218, 228, 238, 248, 258, 268, 278, 288, 298, 308, 318, 328, 338, 348, 358, 368, 378]



ranges = [label_0, label_0, label_0]
list = [average_metadata_no_svd, average_metadata_no_svd_mlp, average_metadata_no_svd_mlp1]
names = ["metadata", "mlp zip 0.8 train", "mlp zip 1.0 train"]

plot_list(ranges, list, names)