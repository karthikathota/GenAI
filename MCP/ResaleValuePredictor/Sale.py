import pandas as pd
import numpy as np
import logging
from typing import Any, Dict, Union
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from mcp.server.fastmcp import FastMCP
from io import StringIO
import re

# Initialize FastMCP
mcp = FastMCP("resale")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dummy CSV data (you can replace this with file loading if needed)
CSV_DATA = """
make,model,year,age,original_price,mileage,condition,estimated_resale_value
Toyota,Corolla,2000,25,35000,416470,Poor,1102
Toyota,Corolla,2001,24,38000,455043,Excellent,1250
Toyota,Corolla,2002,23,40000,339094,Good,1833
Toyota,Corolla,2003,22,42000,370422,Fair,1126
Toyota,Corolla,2004,21,45000,280130,Fair,1592
Toyota,Corolla,2005,20,48000,294417,Excellent,1654
Toyota,Corolla,2006,19,50000,285914,Excellent,1630
Toyota,Corolla,2007,18,52000,229492,Fair,1883
Toyota,Corolla,2008,17,55000,290898,Good,4971
Toyota,Corolla,2009,16,58000,108171,Fair,5852
Toyota,Corolla,2010,15,60000,140887,Fair,4486
Toyota,Corolla,2011,14,65000,133260,Fair,11878
Toyota,Corolla,2012,13,68000,165146,Poor,12859
Toyota,Corolla,2013,12,70000,105821,Excellent,14333
Toyota,Corolla,2014,11,75000,168758,Good,17337
Toyota,Corolla,2015,10,80000,158262,Excellent,18093
Toyota,Corolla,2016,9,85000,140757,Good,17919
Toyota,Corolla,2017,8,90000,48810,Excellent,18094
Toyota,Corolla,2018,7,95000,114802,Fair,17435
Toyota,Corolla,2019,6,100000,31574,Poor,27932
Toyota,Corolla,2020,5,110000,75593,Good,17829
Toyota,Corolla,2021,4,120000,55156,Poor,14252
Toyota,Corolla,2022,3,130000,44377,Good,18252
Toyota,Corolla,2023,2,140000,25763,Fair,19156
Toyota,Camry,2000,25,22017,129240,Excellent,1100
Toyota,Camry,2001,24,30522,147028,Good,1526
Toyota,Camry,2002,23,32068,309197,Excellent,1603
Toyota,Camry,2003,22,38258,124696,Excellent,1912
Toyota,Camry,2004,21,27239,247604,Fair,1361
Toyota,Camry,2005,20,39544,308109,Excellent,1977
Toyota,Camry,2006,19,20523,351982,Excellent,1026
Toyota,Camry,2007,18,35751,151167,Poor,3575
Toyota,Camry,2008,17,26354,255763,Good,3953
Toyota,Camry,2009,16,34438,96350,Good,6887
Toyota,Camry,2010,15,27328,284219,Fair,6832
Toyota,Camry,2011,14,30938,148810,Good,9281
Toyota,Camry,2012,13,30466,160767,Good,10663
Toyota,Camry,2013,12,35929,129960,Fair,14371
Toyota,Camry,2014,11,37577,177914,Poor,16909
Toyota,Camry,2015,10,35895,118713,Poor,17947
Toyota,Camry,2016,9,19143,101645,Excellent,10528
Toyota,Camry,2017,8,39091,48101,Good,23454
Toyota,Camry,2018,7,28331,114219,Excellent,18415
Toyota,Camry,2019,6,15706,95055,Excellent,10994
Toyota,Camry,2020,5,27143,99355,Good,20357
Toyota,Camry,2021,4,29867,62771,Fair,23893
Toyota,Camry,2022,3,20858,46362,Excellent,17729
Toyota,Camry,2023,2,34133,27385,Fair,30719
Toyota,RAV4,2000,25,38933,431198,Poor,1946
Toyota,RAV4,2001,24,20855,324805,Poor,1042
Toyota,RAV4,2002,23,15385,327932,Fair,769
Toyota,RAV4,2003,22,33635,119501,Fair,1681
Toyota,RAV4,2004,21,28213,352520,Excellent,1410
Toyota,RAV4,2005,20,33162,262870,Poor,1658
Toyota,RAV4,2006,19,30836,279468,Poor,1541
Toyota,RAV4,2007,18,22073,342626,Poor,2207
Toyota,RAV4,2008,17,24290,112683,Good,3643
Toyota,RAV4,2009,16,36005,196643,Good,7200
Toyota,RAV4,2010,15,21545,80759,Excellent,5386
Toyota,RAV4,2011,14,26977,178684,Fair,8093
Toyota,RAV4,2012,13,37934,255292,Poor,13276
Toyota,RAV4,2013,12,15628,221693,Excellent,6251
Toyota,RAV4,2014,11,15891,110716,Poor,7150
Toyota,RAV4,2015,10,31176,73815,Fair,15588
Toyota,RAV4,2016,9,15138,56619,Good,8325
Toyota,RAV4,2017,8,37971,159795,Good,22782
Toyota,RAV4,2018,7,27256,64619,Poor,17716
Toyota,RAV4,2019,6,24308,118632,Poor,17015
Toyota,RAV4,2020,5,35372,87312,Good,26529
Toyota,RAV4,2021,4,29383,49505,Fair,23506
Toyota,RAV4,2022,3,19171,28777,Excellent,16295
Toyota,RAV4,2023,2,21620,35573,Excellent,19458
Honda,Civic,2000,25,32575,152407,Fair,1628
Honda,Civic,2001,24,30092,222323,Fair,1504
Honda,Civic,2002,23,23995,247540,Fair,1199
Honda,Civic,2003,22,22476,297278,Excellent,1123
Honda,Civic,2004,21,24847,171834,Fair,1242
Honda,Civic,2005,20,20627,163017,Excellent,1031
Honda,Civic,2006,19,25107,299608,Poor,1255
Honda,Civic,2007,18,35795,250410,Fair,3579
Honda,Civic,2008,17,35424,169195,Fair,5313
Honda,Civic,2009,16,30238,81753,Good,6047
Honda,Civic,2010,15,25065,266331,Good,6266
Honda,Civic,2011,14,17306,137493,Excellent,5191
Honda,Civic,2012,13,38910,201702,Poor,13618
Honda,Civic,2013,12,23238,195942,Excellent,9295
Honda,Civic,2014,11,30605,167881,Excellent,13772
Honda,Civic,2015,10,37724,120705,Poor,18862
Honda,Civic,2016,9,15735,134876,Poor,8654
Honda,Civic,2017,8,38676,135869,Poor,23205
Honda,Civic,2018,7,23859,79676,Fair,15508
Honda,Civic,2019,6,20483,48110,Good,14338
Honda,Civic,2020,5,24384,37355,Fair,18288
Honda,Civic,2021,4,29388,49736,Good,23510
Honda,Civic,2022,3,26088,56426,Good,22174
Honda,Civic,2023,2,15572,19244,Excellent,14014
Honda,Accord,2000,25,36030,486519,Excellent,1801
Honda,Accord,2001,24,30355,349311,Fair,1517
Honda,Accord,2002,23,32245,277212,Excellent,1612
Honda,Accord,2003,22,26967,142919,Fair,1348
Honda,Accord,2004,21,15625,282309,Fair,781
Honda,Accord,2005,20,15273,340268,Excellent,763
Honda,Accord,2006,19,19944,290139,Good,997
Honda,Accord,2007,18,39908,262088,Excellent,3990
Honda,Accord,2008,17,27561,250141,Excellent,4134
Honda,Accord,2009,16,29440,91795,Poor,5887
Honda,Accord,2010,15,38141,196869,Fair,9535
Honda,Accord,2011,14,20590,253964,Poor,6176
Honda,Accord,2012,13,36278,128594,Fair,12697
Honda,Accord,2013,12,28561,92126,Good,11424
Honda,Accord,2014,11,18659,178569,Excellent,8396
Honda,Accord,2015,10,27454,113477,Poor,13727
Honda,Accord,2016,9,17270,167252,Poor,9498
Honda,Accord,2017,8,39557,77555,Fair,23734
Honda,Accord,2018,7,35173,113742,Good,22862
Honda,Accord,2019,6,22162,103051,Good,15513
Honda,Accord,2020,5,32162,73368,Fair,24121
Honda,Accord,2021,4,16547,23087,Poor,13237
Honda,Accord,2022,3,15548,29494,Fair,13215
Honda,Accord,2023,2,36849,11551,Fair,33164
Honda,CR-V,2000,25,29375,292356,Poor,1468
Honda,CR-V,2001,24,31921,415755,Excellent,1596
Honda,CR-V,2002,23,37307,343070,Fair,1865
Honda,CR-V,2003,22,25667,146256,Excellent,1283
Honda,CR-V,2004,21,36014,207652,Poor,1800
Honda,CR-V,2005,20,23298,237150,Excellent,1164
Honda,CR-V,2006,19,24438,169260,Fair,1221
Honda,CR-V,2007,18,38449,220873,Excellent,3844
Honda,CR-V,2008,17,20080,203871,Fair,3011
Honda,CR-V,2009,16,31854,310940,Good,6370
Honda,CR-V,2010,15,22424,171811,Good,5606
Honda,CR-V,2011,14,31962,145602,Fair,9588
Honda,CR-V,2012,13,19351,118556,Fair,6772
Honda,CR-V,2013,12,32622,178015,Poor,13048
Honda,CR-V,2014,11,23591,112608,Good,10615
Honda,CR-V,2015,10,31640,106773,Excellent,15820
Honda,CR-V,2016,9,29823,168891,Good,16402
Honda,CR-V,2017,8,15060,83747,Poor,9036
Honda,CR-V,2018,7,22580,100126,Good,14676
Honda,CR-V,2019,6,18067,70902,Good,12646
Honda,CR-V,2020,5,25045,87731,Poor,18783
Honda,CR-V,2021,4,36745,23056,Excellent,29396
Honda,CR-V,2022,3,20445,56104,Excellent,17378
Honda,CR-V,2023,2,17805,18716,Good,16024
Ford,F-150,2000,25,17692,411910,Excellent,884
Ford,F-150,2001,24,18023,334533,Poor,901
Ford,F-150,2002,23,15976,152196,Poor,798
Ford,F-150,2003,22,37176,133153,Excellent,1858
Ford,F-150,2004,21,31611,314069,Poor,1580
Ford,F-150,2005,20,36769,144520,Poor,1838
Ford,F-150,2006,19,33012,340059,Excellent,1650
Ford,F-150,2007,18,36253,154237,Poor,3625
Ford,F-150,2008,17,17510,312770,Fair,2626
Ford,F-150,2009,16,24628,214862,Excellent,4925
Ford,F-150,2010,15,27869,292827,Fair,6967
Ford,F-150,2011,14,25170,172964,Poor,7550
Ford,F-150,2012,13,31024,222461,Poor,10858
Ford,F-150,2013,12,25892,149958,Good,10356
Ford,F-150,2014,11,30995,205707,Good,13947
Ford,F-150,2015,10,27595,92018,Excellent,13797
Ford,F-150,2016,9,30991,179780,Excellent,17045
Ford,F-150,2017,8,36444,112404,Poor,21866
Ford,F-150,2018,7,35709,127823,Excellent,23210
Ford,F-150,2019,6,15925,31366,Fair,11147
Ford,F-150,2020,5,27200,44717,Poor,20400
Ford,F-150,2021,4,26281,57311,Fair,21024
Ford,F-150,2022,3,20418,35878,Good,17355
Ford,F-150,2023,2,28193,19998,Good,25373
Ford,Focus,2000,25,25452,181572,Excellent,1272
Ford,Focus,2001,24,27453,469852,Good,1372
Ford,Focus,2002,23,37512,339491,Fair,1875
Ford,Focus,2003,22,35228,243138,Excellent,1761
Ford,Focus,2004,21,18884,273061,Good,944
Ford,Focus,2005,20,25181,117714,Fair,1259
Ford,Focus,2006,19,27190,359161,Poor,1359
Ford,Focus,2007,18,15384,247891,Poor,1538
Ford,Focus,2008,17,28247,128577,Fair,4237
Ford,Focus,2009,16,30953,201374,Fair,6190
Ford,Focus,2010,15,22751,133364,Poor,5687
Ford,Focus,2011,14,34335,156589,Fair,10300
Ford,Focus,2012,13,34053,190865,Excellent,11918
Ford,Focus,2013,12,35964,214263,Excellent,14385
Ford,Focus,2014,11,15201,61992,Good,6840
Ford,Focus,2015,10,19121,186452,Fair,9560
Ford,Focus,2016,9,34849,58749,Excellent,19166
Ford,Focus,2017,8,25976,66298,Poor,15585
Ford,Focus,2018,7,28136,97556,Excellent,18288
Ford,Focus,2019,6,25878,67406,Excellent,18114
Ford,Focus,2020,5,19557,29292,Poor,14667
Ford,Focus,2021,4,21749,40185,Fair,17399
Ford,Focus,2022,3,27930,28943,Excellent,23740
Ford,Focus,2023,2,21147,23717,Fair,19032
Ford,Escape,2000,25,18606,334730,Fair,930
Ford,Escape,2001,24,27820,156049,Excellent,1391
Ford,Escape,2002,23,34957,282870,Fair,1747
Ford,Escape,2003,22,38909,379147,Good,1945
Ford,Escape,2004,21,16175,140446,Fair,808
Ford,Escape,2005,20,27058,399546,Excellent,1352
Ford,Escape,2006,19,39586,111412,Fair,1979
Ford,Escape,2007,18,18586,327692,Good,1858
Ford,Escape,2008,17,23998,121028,Fair,3599
Ford,Escape,2009,16,20731,222495,Excellent,4146
Ford,Escape,2010,15,25764,228185,Poor,6441
Ford,Escape,2011,14,26775,151910,Excellent,8032
Ford,Escape,2012,13,30628,176910,Poor,10719
Ford,Escape,2013,12,16930,80645,Good,6771
Ford,Escape,2014,11,38467,181162,Fair,17310
Ford,Escape,2015,10,26269,109314,Fair,13134
Ford,Escape,2016,9,15228,177177,Fair,8375
Ford,Escape,2017,8,32218,152913,Fair,19330
Ford,Escape,2018,7,17765,63926,Good,11547
Ford,Escape,2019,6,19577,50044,Excellent,13703
Ford,Escape,2020,5,31338,63870,Fair,23503
Ford,Escape,2021,4,39513,22331,Good,31610
Ford,Escape,2022,3,36815,54212,Good,31292
Ford,Escape,2023,2,34018,34856,Excellent,30616
Chevrolet,Silverado,2000,25,18956,427242,Poor,947
Chevrolet,Silverado,2001,24,15552,335615,Excellent,777
Chevrolet,Silverado,2002,23,18615,364862,Fair,930
Chevrolet,Silverado,2003,22,22019,249149,Poor,1100
Chevrolet,Silverado,2004,21,20681,288866,Excellent,1034
Chevrolet,Silverado,2005,20,19183,168860,Fair,959
Chevrolet,Silverado,2006,19,27528,238801,Fair,1376
Chevrolet,Silverado,2007,18,24191,338740,Fair,2419
Chevrolet,Silverado,2008,17,23239,206602,Excellent,3485
Chevrolet,Silverado,2009,16,25032,294979,Fair,5006
Chevrolet,Silverado,2010,15,15807,249930,Poor,3951
Chevrolet,Silverado,2011,14,24480,117757,Excellent,7343
Chevrolet,Silverado,2012,13,22007,162645,Fair,7702
Chevrolet,Silverado,2013,12,37937,223315,Good,15174
Chevrolet,Silverado,2014,11,15660,74647,Excellent,7046
Chevrolet,Silverado,2015,10,37598,182733,Excellent,18799
Chevrolet,Silverado,2016,9,32708,129789,Fair,17989
Chevrolet,Silverado,2017,8,36369,110476,Good,21821
Chevrolet,Silverado,2018,7,32689,106221,Good,21247
Chevrolet,Silverado,2019,6,34735,83405,Poor,24314
Chevrolet,Silverado,2020,5,24011,48532,Good,18008
Chevrolet,Silverado,2021,4,25484,51708,Excellent,20387
Chevrolet,Silverado,2022,3,28634,36121,Fair,24338
Chevrolet,Silverado,2023,2,35052,27664,Good,31546
Chevrolet,Malibu,2000,25,29981,169341,Excellent,1499
Chevrolet,Malibu,2001,24,33695,339007,Excellent,1684
Chevrolet,Malibu,2002,23,15934,446758,Excellent,796
Chevrolet,Malibu,2003,22,24272,356302,Fair,1213
Chevrolet,Malibu,2004,21,18997,372420,Excellent,949
Chevrolet,Malibu,2005,20,36505,301072,Excellent,1825
Chevrolet,Malibu,2006,19,20668,260035,Excellent,1033
Chevrolet,Malibu,2007,18,19571,340737,Fair,1957
Chevrolet,Malibu,2008,17,19476,202824,Fair,2921
Chevrolet,Malibu,2009,16,38535,263623,Excellent,7706
Chevrolet,Malibu,2010,15,37777,149802,Excellent,9444
Chevrolet,Malibu,2011,14,16641,258361,Fair,4992
Chevrolet,Malibu,2012,13,37499,252204,Poor,13124
Chevrolet,Malibu,2013,12,28190,71148,Good,11275
Chevrolet,Malibu,2014,11,25938,154837,Good,11672
Chevrolet,Malibu,2015,10,30340,170592,Good,15170
Chevrolet,Malibu,2016,9,38970,121237,Fair,21433
Chevrolet,Malibu,2017,8,35882,127985,Good,21529
Chevrolet,Malibu,2018,7,19388,86615,Fair,12602
Chevrolet,Malibu,2019,6,37823,42944,Poor,26476
Chevrolet,Malibu,2020,5,21025,88442,Excellent,15768
Chevrolet,Malibu,2021,4,36195,52015,Fair,28956
Chevrolet,Malibu,2022,3,15444,50608,Good,13127
Chevrolet,Malibu,2023,2,34757,11779,Good,31281
Chevrolet,Equinox,2000,25,32928,326816,Excellent,1646
Chevrolet,Equinox,2001,24,27525,446342,Fair,1376
Chevrolet,Equinox,2002,23,29136,259404,Poor,1456
Chevrolet,Equinox,2003,22,24909,118349,Poor,1245
Chevrolet,Equinox,2004,21,32084,117387,Fair,1604
Chevrolet,Equinox,2005,20,38982,151601,Poor,1949
Chevrolet,Equinox,2006,19,39204,256564,Good,1960
Chevrolet,Equinox,2007,18,26237,250370,Poor,2623
Chevrolet,Equinox,2008,17,31625,135781,Fair,4743
Chevrolet,Equinox,2009,16,36585,94232,Good,7316
Chevrolet,Equinox,2010,15,28215,175137,Fair,7053
Chevrolet,Equinox,2011,14,24842,97585,Excellent,7452
Chevrolet,Equinox,2012,13,16695,242911,Poor,5843
Chevrolet,Equinox,2013,12,23786,137755,Fair,9514
Chevrolet,Equinox,2014,11,33328,99177,Poor,14997
Chevrolet,Equinox,2015,10,28046,161703,Poor,14023
Chevrolet,Equinox,2016,9,23120,83152,Fair,12716
Chevrolet,Equinox,2017,8,24197,122632,Excellent,14518
Chevrolet,Equinox,2018,7,27188,112501,Excellent,17672
Chevrolet,Equinox,2019,6,28931,102053,Poor,20251
Chevrolet,Equinox,2020,5,18071,31991,Excellent,13553
Chevrolet,Equinox,2021,4,37511,36607,Fair,30008
Chevrolet,Equinox,2022,3,17841,32946,Excellent,15164
Chevrolet,Equinox,2023,2,24328,12200,Poor,21895
Nissan,Altima,2000,25,19923,498352,Poor,996
Nissan,Altima,2001,24,34738,298835,Good,1736
Nissan,Altima,2002,23,16646,148581,Good,832
Nissan,Altima,2003,22,38945,116142,Poor,1947
Nissan,Altima,2004,21,25948,203950,Good,1297
Nissan,Altima,2005,20,28965,348778,Poor,1448
Nissan,Altima,2006,19,38690,250459,Excellent,1934
Nissan,Altima,2007,18,29139,224235,Poor,2913
Nissan,Altima,2008,17,15659,133699,Good,2348
Nissan,Altima,2009,16,39904,92976,Excellent,7980
Nissan,Altima,2010,15,16003,215577,Fair,4000
Nissan,Altima,2011,14,20477,96272,Poor,6143
Nissan,Altima,2012,13,24528,133803,Poor,8584
Nissan,Altima,2013,12,17868,183194,Poor,7147
Nissan,Altima,2014,11,39362,125189,Fair,17712
Nissan,Altima,2015,10,24755,181876,Fair,12377
Nissan,Altima,2016,9,18149,68732,Poor,9981
Nissan,Altima,2017,8,24355,60659,Excellent,14613
Nissan,Altima,2018,7,32354,78051,Fair,21030
Nissan,Altima,2019,6,36209,31990,Fair,25346
Nissan,Altima,2020,5,19108,38340,Poor,14331
Nissan,Altima,2021,4,33149,40371,Good,26519
Nissan,Altima,2022,3,38209,27807,Fair,32477
Nissan,Altima,2023,2,18189,35693,Good,16370
Nissan,Sentra,2000,25,27757,141505,Poor,1387
Nissan,Sentra,2001,24,26705,192823,Good,1335
Nissan,Sentra,2002,23,19263,208584,Good,963
Nissan,Sentra,2003,22,36541,296360,Fair,1827
Nissan,Sentra,2004,21,20581,108285,Fair,1029
Nissan,Sentra,2005,20,15684,161345,Good,784
Nissan,Sentra,2006,19,15850,187258,Good,792
Nissan,Sentra,2007,18,31492,194367,Poor,3149
Nissan,Sentra,2008,17,20267,116274,Excellent,3040
Nissan,Sentra,2009,16,25869,223133,Fair,5173
Nissan,Sentra,2010,15,17694,219992,Good,4423
Nissan,Sentra,2011,14,17632,169312,Good,5289
Nissan,Sentra,2012,13,27413,106536,Poor,9594
Nissan,Sentra,2013,12,35615,211629,Fair,14245
Nissan,Sentra,2014,11,38874,154500,Excellent,17493
Nissan,Sentra,2015,10,38640,117179,Poor,19320
Nissan,Sentra,2016,9,21747,85902,Good,11960
Nissan,Sentra,2017,8,27760,143932,Excellent,16656
Nissan,Sentra,2018,7,16976,68539,Fair,11034
Nissan,Sentra,2019,6,29086,116302,Fair,20360
Nissan,Sentra,2020,5,35279,58424,Poor,26459
Nissan,Sentra,2021,4,22712,67431,Poor,18169
Nissan,Sentra,2022,3,38085,45701,Good,32372
Nissan,Sentra,2023,2,34912,13294,Good,31420
Nissan,Rogue,2000,25,36832,282476,Poor,1841
Nissan,Rogue,2001,24,37080,221103,Fair,1854
Nissan,Rogue,2002,23,21172,438564,Good,1058
Nissan,Rogue,2003,22,36460,420539,Poor,1823
Nissan,Rogue,2004,21,15747,293271,Good,787
Nissan,Rogue,2005,20,33482,125008,Fair,1674
Nissan,Rogue,2006,19,31782,101688,Good,1589
Nissan,Rogue,2007,18,21122,171724,Good,2112
Nissan,Rogue,2008,17,35089,96007,Poor,5263
Nissan,Rogue,2009,16,18725,302125,Excellent,3744
Nissan,Rogue,2010,15,24988,294410,Poor,6247
Nissan,Rogue,2011,14,29708,79002,Poor,8912
Nissan,Rogue,2012,13,24798,125966,Fair,8679
Nissan,Rogue,2013,12,17334,95884,Poor,6933
Nissan,Rogue,2014,11,25823,56033,Fair,11620
Nissan,Rogue,2015,10,31973,140798,Fair,15986
Nissan,Rogue,2016,9,31887,95299,Fair,17537
Nissan,Rogue,2017,8,30493,102569,Excellent,18295
Nissan,Rogue,2018,7,19155,38343,Good,12450
Nissan,Rogue,2019,6,25346,37227,Fair,17742
Nissan,Rogue,2020,5,15897,71696,Poor,11922
Nissan,Rogue,2021,4,38141,71568,Excellent,30512
Nissan,Rogue,2022,3,26763,52168,Good,22748
Nissan,Rogue,2023,2,35467,32870,Fair,31920
BMW,3 Series,2000,25,20043,232122,Excellent,1002
BMW,3 Series,2001,24,15633,181299,Poor,781
BMW,3 Series,2002,23,20193,210864,Fair,1009
BMW,3 Series,2003,22,36824,265794,Poor,1841
BMW,3 Series,2004,21,30115,228924,Fair,1505
BMW,3 Series,2005,20,29872,335908,Excellent,1493
BMW,3 Series,2006,19,24647,103580,Good,1232
BMW,3 Series,2007,18,15300,353419,Good,1529
BMW,3 Series,2008,17,31166,172966,Excellent,4674
BMW,3 Series,2009,16,15928,309414,Good,3185
BMW,3 Series,2010,15,26906,107284,Poor,6726
BMW,3 Series,2011,14,37540,234251,Poor,11261
BMW,3 Series,2012,13,37646,127463,Good,13176
BMW,3 Series,2013,12,28164,132392,Excellent,11265
BMW,3 Series,2014,11,37388,67485,Poor,16824
BMW,3 Series,2015,10,36348,133399,Fair,18174
BMW,3 Series,2016,9,24274,101039,Excellent,13350
BMW,3 Series,2017,8,26986,153082,Excellent,16191
BMW,3 Series,2018,7,17055,120379,Good,11085
BMW,3 Series,2019,6,18024,36540,Excellent,12616
BMW,3 Series,2020,5,26881,70803,Excellent,20160
BMW,3 Series,2021,4,26697,69561,Good,21357
BMW,3 Series,2022,3,34927,40653,Fair,29687
BMW,3 Series,2023,2,21520,30632,Good,19368
BMW,5 Series,2000,25,21783,293495,Excellent,1089
BMW,5 Series,2001,24,32706,232770,Excellent,1635
BMW,5 Series,2002,23,21776,389605,Excellent,1088
BMW,5 Series,2003,22,32609,318193,Excellent,1630
BMW,5 Series,2004,21,22822,114123,Fair,1141
BMW,5 Series,2005,20,38778,367690,Good,1938
BMW,5 Series,2006,19,34866,227680,Excellent,1743
BMW,5 Series,2007,18,34827,271021,Poor,3482
BMW,5 Series,2008,17,38936,106942,Poor,5840
BMW,5 Series,2009,16,30755,239475,Poor,6150
BMW,5 Series,2010,15,26843,100031,Excellent,6710
BMW,5 Series,2011,14,38903,248695,Fair,11670
BMW,5 Series,2012,13,38668,151898,Good,13533
BMW,5 Series,2013,12,25539,180534,Good,10215
BMW,5 Series,2014,11,23723,96793,Good,10675
BMW,5 Series,2015,10,20163,197844,Good,10081
BMW,5 Series,2016,9,24699,115619,Excellent,13584
BMW,5 Series,2017,8,20169,154081,Good,12101
BMW,5 Series,2018,7,16728,111167,Fair,10873
BMW,5 Series,2019,6,18354,37652,Fair,12847
BMW,5 Series,2020,5,38701,84205,Fair,29025
BMW,5 Series,2021,4,18030,41542,Poor,14424
BMW,5 Series,2022,3,35291,57127,Fair,29997
BMW,5 Series,2023,2,15786,26320,Good,14207
BMW,X5,2000,25,20747,325306,Good,1037
BMW,X5,2001,24,31028,455384,Fair,1551
BMW,X5,2002,23,20461,408956,Good,1023
BMW,X5,2003,22,27728,334291,Poor,1386
BMW,X5,2004,21,38896,262591,Excellent,1944
BMW,X5,2005,20,25340,384884,Good,1267
BMW,X5,2006,19,31338,221467,Fair,1566
BMW,X5,2007,18,39721,122019,Excellent,3972
BMW,X5,2008,17,31145,115340,Poor,4671
BMW,X5,2009,16,19770,268111,Excellent,3953
BMW,X5,2010,15,24268,158334,Poor,6067
BMW,X5,2011,14,23134,251831,Excellent,6940
BMW,X5,2012,13,27017,259773,Fair,9455
BMW,X5,2013,12,35558,76536,Excellent,14223
BMW,X5,2014,11,16706,96938,Fair,7517
BMW,X5,2015,10,30621,94646,Good,15310
BMW,X5,2016,9,25929,89617,Excellent,14260
BMW,X5,2017,8,34023,123926,Good,20413
BMW,X5,2018,7,30678,40520,Poor,19940
BMW,X5,2019,6,24325,85492,Good,17027
BMW,X5,2020,5,34242,25767,Fair,25681
BMW,X5,2021,4,21685,23131,Good,17348
BMW,X5,2022,3,16634,55664,Poor,14138
BMW,X5,2023,2,17314,12998,Good,15582
Mercedes,C-Class,2000,25,18303,381981,Poor,915
Mercedes,C-Class,2001,24,25392,186529,Poor,1269
Mercedes,C-Class,2002,23,33911,279179,Poor,1695
Mercedes,C-Class,2003,22,34237,376397,Poor,1711
Mercedes,C-Class,2004,21,30870,186627,Poor,1543
Mercedes,C-Class,2005,20,33636,277333,Poor,1681
Mercedes,C-Class,2006,19,26694,350136,Fair,1334
Mercedes,C-Class,2007,18,30397,300434,Poor,3039
Mercedes,C-Class,2008,17,23057,312233,Good,3458
Mercedes,C-Class,2009,16,36784,288013,Excellent,7356
Mercedes,C-Class,2010,15,22423,266148,Good,5605
Mercedes,C-Class,2011,14,17477,271194,Fair,5243
Mercedes,C-Class,2012,13,39191,201997,Good,13716
Mercedes,C-Class,2013,12,28581,209328,Fair,11432
Mercedes,C-Class,2014,11,27710,76760,Excellent,12469
Mercedes,C-Class,2015,10,21446,145795,Fair,10723
Mercedes,C-Class,2016,9,34605,100538,Good,19032
Mercedes,C-Class,2017,8,29211,135798,Poor,17526
Mercedes,C-Class,2018,7,25178,102646,Excellent,16365
Mercedes,C-Class,2019,6,33636,93387,Poor,23545
Mercedes,C-Class,2020,5,25493,72616,Fair,19119
Mercedes,C-Class,2021,4,36946,27029,Good,29556
Mercedes,C-Class,2022,3,22466,33150,Fair,19096
Mercedes,C-Class,2023,2,27351,14335,Poor,24615
Mercedes,E-Class,2000,25,31109,302660,Excellent,1555
Mercedes,E-Class,2001,24,15912,304923,Poor,795
Mercedes,E-Class,2002,23,19651,357484,Good,982
Mercedes,E-Class,2003,22,33102,229820,Good,1655
Mercedes,E-Class,2004,21,39800,394573,Excellent,1990
Mercedes,E-Class,2005,20,39087,391235,Poor,1954
Mercedes,E-Class,2006,19,15934,278322,Good,796
Mercedes,E-Class,2007,18,32013,129224,Excellent,3201
Mercedes,E-Class,2008,17,35842,181841,Fair,5376
Mercedes,E-Class,2009,16,23816,209963,Fair,4763
Mercedes,E-Class,2010,15,28558,242379,Poor,7139
Mercedes,E-Class,2011,14,39745,241359,Good,11923
Mercedes,E-Class,2012,13,26317,151503,Poor,9210
Mercedes,E-Class,2013,12,22238,80548,Good,8895
Mercedes,E-Class,2014,11,19585,154312,Excellent,8813
Mercedes,E-Class,2015,10,20695,186489,Fair,10347
Mercedes,E-Class,2016,9,27268,99078,Fair,14997
Mercedes,E-Class,2017,8,35323,44953,Excellent,21193
Mercedes,E-Class,2018,7,31659,37724,Good,20578
Mercedes,E-Class,2019,6,19755,86114,Excellent,13828
Mercedes,E-Class,2020,5,29954,66932,Fair,22465
Mercedes,E-Class,2021,4,16274,51848,Fair,13019
Mercedes,E-Class,2022,3,36821,49043,Poor,31297
Mercedes,E-Class,2023,2,21290,28210,Excellent,19161
Mercedes,GLC,2000,25,28495,337974,Good,1424
Mercedes,GLC,2001,24,15530,147076,Excellent,776
Mercedes,GLC,2002,23,27151,209765,Excellent,1357
Mercedes,GLC,2003,22,16326,422720,Excellent,816
Mercedes,GLC,2004,21,33361,397417,Excellent,1668
Mercedes,GLC,2005,20,22532,129047,Good,1126
Mercedes,GLC,2006,19,21836,349836,Fair,1091
Mercedes,GLC,2007,18,34476,284085,Fair,3447
Mercedes,GLC,2008,17,31162,115706,Fair,4674
Mercedes,GLC,2009,16,39670,316727,Fair,7933
Mercedes,GLC,2010,15,34170,209909,Fair,8542
Mercedes,GLC,2011,14,20646,194514,Good,6193
Mercedes,GLC,2012,13,23517,107480,Fair,8230
Mercedes,GLC,2013,12,26175,68943,Excellent,10469
Mercedes,GLC,2014,11,39675,196522,Good,17853
Mercedes,GLC,2015,10,26812,187832,Good,13406
Mercedes,GLC,2016,9,21630,176742,Fair,11896
Mercedes,GLC,2017,8,38223,66199,Fair,22933
Mercedes,GLC,2018,7,16520,35405,Excellent,10737
Mercedes,GLC,2019,6,17911,58837,Fair,12537
Mercedes,GLC,2020,5,34221,83336,Poor,25665
Mercedes,GLC,2021,4,15491,33964,Poor,12392
Mercedes,GLC,2022,3,27766,17277,Good,23601
Mercedes,GLC,2023,2,35985,36222,Good,32386
Hyundai,Elantra,2000,25,20511,494572,Poor,1025
Hyundai,Elantra,2001,24,23386,136022,Fair,1169
Hyundai,Elantra,2002,23,17259,259290,Fair,862
Hyundai,Elantra,2003,22,21663,149723,Good,1083
Hyundai,Elantra,2004,21,29908,262273,Poor,1495
Hyundai,Elantra,2005,20,15780,267896,Excellent,789
Hyundai,Elantra,2006,19,31394,321277,Poor,1569
Hyundai,Elantra,2007,18,17333,165920,Fair,1733
Hyundai,Elantra,2008,17,16805,150803,Fair,2520
Hyundai,Elantra,2009,16,31902,107222,Excellent,6380
Hyundai,Elantra,2010,15,32577,299101,Poor,8144
Hyundai,Elantra,2011,14,35561,123143,Poor,10668
Hyundai,Elantra,2012,13,28388,137010,Excellent,9935
Hyundai,Elantra,2013,12,28259,91301,Fair,11303
Hyundai,Elantra,2014,11,30738,122830,Poor,13832
Hyundai,Elantra,2015,10,32874,157453,Excellent,16437
Hyundai,Elantra,2016,9,26897,50522,Fair,14793
Hyundai,Elantra,2017,8,19032,63465,Excellent,11419
Hyundai,Elantra,2018,7,15565,115984,Excellent,10117
Hyundai,Elantra,2019,6,15461,111665,Fair,10822
Hyundai,Elantra,2020,5,31887,92040,Excellent,23915
Hyundai,Elantra,2021,4,16885,45170,Fair,13508
Hyundai,Elantra,2022,3,25600,15985,Fair,21760
Hyundai,Elantra,2023,2,38983,10139,Good,35084
Hyundai,Sonata,2000,25,39240,286718,Fair,1962
Hyundai,Sonata,2001,24,33961,207725,Excellent,1698
Hyundai,Sonata,2002,23,31504,406008,Good,1575
Hyundai,Sonata,2003,22,34000,183369,Poor,1700
Hyundai,Sonata,2004,21,33949,140554,Excellent,1697
Hyundai,Sonata,2005,20,33234,229689,Poor,1661
Hyundai,Sonata,2006,19,24586,286092,Excellent,1229
Hyundai,Sonata,2007,18,29638,167114,Excellent,2963
Hyundai,Sonata,2008,17,30269,240582,Poor,4540
Hyundai,Sonata,2009,16,24635,212719,Excellent,4926
Hyundai,Sonata,2010,15,34579,186664,Good,8644
Hyundai,Sonata,2011,14,34097,138905,Fair,10229
Hyundai,Sonata,2012,13,31368,198019,Fair,10978
Hyundai,Sonata,2013,12,23898,179476,Good,9559
Hyundai,Sonata,2014,11,33728,153808,Fair,15177
Hyundai,Sonata,2015,10,18995,141006,Good,9497
Hyundai,Sonata,2016,9,36597,85239,Excellent,20128
Hyundai,Sonata,2017,8,28398,63202,Poor,17038
Hyundai,Sonata,2018,7,15307,135803,Fair,9949
Hyundai,Sonata,2019,6,34154,118724,Good,23907
Hyundai,Sonata,2020,5,32920,75473,Fair,24690
Hyundai,Sonata,2021,4,37188,30175,Excellent,29750
Hyundai,Sonata,2022,3,21055,42193,Good,17896
Hyundai,Sonata,2023,2,19013,30961,Fair,17111
Hyundai,Tucson,2000,25,27938,255834,Poor,1396
Hyundai,Tucson,2001,24,39522,208545,Excellent,1976
Hyundai,Tucson,2002,23,32662,338955,Fair,1633
Hyundai,Tucson,2003,22,35369,231205,Fair,1768
Hyundai,Tucson,2004,21,15727,159952,Good,786
Hyundai,Tucson,2005,20,22191,347603,Poor,1109
Hyundai,Tucson,2006,19,17327,135245,Excellent,866
Hyundai,Tucson,2007,18,38089,149939,Poor,3808
Hyundai,Tucson,2008,17,28978,144540,Excellent,4346
Hyundai,Tucson,2009,16,22055,267992,Fair,4410
Hyundai,Tucson,2010,15,21961,286204,Fair,5490
Hyundai,Tucson,2011,14,18655,174678,Poor,5596
Hyundai,Tucson,2012,13,20744,205874,Fair,7260
Hyundai,Tucson,2013,12,24248,149582,Excellent,9699
Hyundai,Tucson,2014,11,29265,109983,Poor,13169
Hyundai,Tucson,2015,10,29941,80265,Good,14970
Hyundai,Tucson,2016,9,32685,60408,Poor,17976
Hyundai,Tucson,2017,8,29661,116506,Good,17796
Hyundai,Tucson,2018,7,34014,132767,Fair,22109
Hyundai,Tucson,2019,6,39146,74478,Good,27402
Hyundai,Tucson,2020,5,31696,87847,Poor,23772
Hyundai,Tucson,2021,4,33385,55470,Good,26708
Hyundai,Tucson,2022,3,24521,26333,Fair,20842
Hyundai,Tucson,2023,2,22365,39324,Good,20128
Kia,Forte,2000,25,31845,450388,Poor,1592
Kia,Forte,2001,24,17237,251842,Good,861
Kia,Forte,2002,23,33061,409423,Fair,1653
Kia,Forte,2003,22,36569,281045,Poor,1828
Kia,Forte,2004,21,32972,116029,Good,1648
Kia,Forte,2005,20,18469,281471,Excellent,923
Kia,Forte,2006,19,30067,173759,Fair,1503
Kia,Forte,2007,18,16260,254762,Fair,1625
Kia,Forte,2008,17,19865,207876,Poor,2979
Kia,Forte,2009,16,31104,226850,Excellent,6220
Kia,Forte,2010,15,38266,122768,Poor,9566
Kia,Forte,2011,14,35647,252992,Fair,10694
Kia,Forte,2012,13,38255,187035,Fair,13389
Kia,Forte,2013,12,25581,60892,Poor,10232
Kia,Forte,2014,11,32738,93115,Poor,14732
Kia,Forte,2015,10,25094,181753,Fair,12547
Kia,Forte,2016,9,23657,138815,Good,13011
Kia,Forte,2017,8,28830,74725,Poor,17298
Kia,Forte,2018,7,15708,137588,Fair,10210
Kia,Forte,2019,6,15563,89309,Excellent,10894
Kia,Forte,2020,5,26551,91293,Excellent,19913
Kia,Forte,2021,4,37197,34389,Fair,29757
Kia,Forte,2022,3,33668,48790,Poor,28617
Kia,Forte,2023,2,19896,32437,Fair,17906
Kia,Optima,2000,25,18349,146289,Poor,917
Kia,Optima,2001,24,33062,332813,Poor,1653
Kia,Optima,2002,23,16105,392173,Good,805
Kia,Optima,2003,22,30805,140016,Excellent,1540
Kia,Optima,2004,21,21122,345044,Excellent,1056
Kia,Optima,2005,20,27080,210507,Good,1354
Kia,Optima,2006,19,29056,136419,Fair,1452
Kia,Optima,2007,18,32183,316626,Good,3218
Kia,Optima,2008,17,37930,236160,Excellent,5689
Kia,Optima,2009,16,25290,104676,Good,5057
Kia,Optima,2010,15,34120,82049,Poor,8530
Kia,Optima,2011,14,16514,249323,Poor,4954
Kia,Optima,2012,13,23132,226650,Poor,8096
Kia,Optima,2013,12,24969,94571,Excellent,9987
Kia,Optima,2014,11,18839,62817,Fair,8477
Kia,Optima,2015,10,36678,76947,Excellent,18339
Kia,Optima,2016,9,24021,116666,Excellent,13211
Kia,Optima,2017,8,39424,134012,Poor,23654
Kia,Optima,2018,7,17534,42202,Excellent,11397
Kia,Optima,2019,6,34511,119726,Poor,24157
Kia,Optima,2020,5,35007,66100,Good,26255
Kia,Optima,2021,4,33039,51358,Good,26431
Kia,Optima,2022,3,29498,24391,Good,25073
Kia,Optima,2023,2,21988,18706,Good,19789
Kia,Sportage,2000,25,20930,464033,Good,1046
Kia,Sportage,2001,24,19732,139980,Good,986
Kia,Sportage,2002,23,38797,120254,Good,1939
Kia,Sportage,2003,22,33377,296997,Poor,1668
Kia,Sportage,2004,21,38270,368586,Fair,1913
Kia,Sportage,2005,20,22497,321733,Excellent,1124
Kia,Sportage,2006,19,22865,194164,Good,1143
Kia,Sportage,2007,18,39258,308319,Excellent,3925
Kia,Sportage,2008,17,29613,238750,Poor,4441
Kia,Sportage,2009,16,23005,306736,Poor,4600
Kia,Sportage,2010,15,39951,222307,Good,9987
Kia,Sportage,2011,14,24844,247561,Fair,7453
Kia,Sportage,2012,13,32652,193366,Fair,11428
Kia,Sportage,2013,12,15341,210027,Poor,6136
Kia,Sportage,2014,11,20363,105611,Fair,9163
Kia,Sportage,2015,10,26181,161648,Poor,13090
Kia,Sportage,2016,9,39869,53870,Fair,21927
Kia,Sportage,2017,8,30759,99328,Fair,18455
Kia,Sportage,2018,7,23312,83194,Fair,15152
Kia,Sportage,2019,6,35209,63356,Fair,24646
Kia,Sportage,2020,5,29513,27902,Fair,22134
Kia,Sportage,2021,4,19626,41336,Excellent,15700
Kia,Sportage,2022,3,33643,25091,Good,28596
Kia,Sportage,2023,2,36644,33015,Excellent,32979
Jeep,Wrangler,2000,25,28108,446019,Fair,1405
Jeep,Wrangler,2001,24,23786,219806,Good,1189
Jeep,Wrangler,2002,23,38450,302465,Excellent,1922
Jeep,Wrangler,2003,22,25220,386418,Poor,1261
Jeep,Wrangler,2004,21,38125,412574,Fair,1906
Jeep,Wrangler,2005,20,29006,373075,Poor,1450
Jeep,Wrangler,2006,19,30721,361094,Excellent,1536
Jeep,Wrangler,2007,18,34731,341885,Excellent,3473
Jeep,Wrangler,2008,17,15901,256317,Good,2385
Jeep,Wrangler,2009,16,32687,100752,Good,6537
Jeep,Wrangler,2010,15,34985,92287,Poor,8746
Jeep,Wrangler,2011,14,19832,247401,Fair,5949
Jeep,Wrangler,2012,13,17760,158658,Good,6216
Jeep,Wrangler,2013,12,29423,152849,Poor,11769
Jeep,Wrangler,2014,11,21109,196434,Fair,9499
Jeep,Wrangler,2015,10,26184,82123,Good,13092
Jeep,Wrangler,2016,9,31242,129433,Good,17183
Jeep,Wrangler,2017,8,24127,137304,Excellent,14476
Jeep,Wrangler,2018,7,24925,125200,Fair,16201
Jeep,Wrangler,2019,6,31485,41388,Fair,22039
Jeep,Wrangler,2020,5,35513,33441,Excellent,26634
Jeep,Wrangler,2021,4,36032,50163,Fair,28825
Jeep,Wrangler,2022,3,31384,40123,Excellent,26676
Jeep,Wrangler,2023,2,24410,34797,Poor,21969
Jeep,Cherokee,2000,25,32983,296729,Excellent,1649
Jeep,Cherokee,2001,24,34429,141147,Excellent,1721
Jeep,Cherokee,2002,23,16660,190168,Poor,833
Jeep,Cherokee,2003,22,31675,309783,Poor,1583
Jeep,Cherokee,2004,21,35831,397765,Good,1791
Jeep,Cherokee,2005,20,23295,115935,Excellent,1164
Jeep,Cherokee,2006,19,39187,175119,Fair,1959
Jeep,Cherokee,2007,18,24690,92073,Poor,2468
Jeep,Cherokee,2008,17,29820,145542,Good,4472
Jeep,Cherokee,2009,16,34048,102630,Fair,6809
Jeep,Cherokee,2010,15,21685,144540,Poor,5421
Jeep,Cherokee,2011,14,17929,125526,Excellent,5378
Jeep,Cherokee,2012,13,30714,244416,Poor,10749
Jeep,Cherokee,2013,12,32632,195270,Excellent,13052
Jeep,Cherokee,2014,11,27554,214588,Poor,12399
Jeep,Cherokee,2015,10,38082,196266,Fair,19041
Jeep,Cherokee,2016,9,36821,102877,Fair,20251
Jeep,Cherokee,2017,8,30142,75636,Excellent,18085
Jeep,Cherokee,2018,7,36556,134411,Good,23761
Jeep,Cherokee,2019,6,38724,96455,Good,27106
Jeep,Cherokee,2020,5,24860,62575,Excellent,18645
Jeep,Cherokee,2021,4,27549,28211,Excellent,22039
Jeep,Cherokee,2022,3,30384,32402,Good,25826
Jeep,Cherokee,2023,2,19589,32970,Fair,17630
Jeep,Grand Cherokee,2000,25,20413,148285,Fair,1020
Jeep,Grand Cherokee,2001,24,15297,181393,Good,764
Jeep,Grand Cherokee,2002,23,25566,320823,Excellent,1278
Jeep,Grand Cherokee,2003,22,20941,263207,Poor,1047
Jeep,Grand Cherokee,2004,21,32711,338470,Poor,1635
Jeep,Grand Cherokee,2005,20,39182,227210,Fair,1959
Jeep,Grand Cherokee,2006,19,39013,193831,Fair,1950
Jeep,Grand Cherokee,2007,18,31448,272869,Fair,3144
Jeep,Grand Cherokee,2008,17,36138,86943,Poor,5420
Jeep,Grand Cherokee,2009,16,39569,126838,Poor,7913
Jeep,Grand Cherokee,2010,15,34006,237449,Excellent,8501
Jeep,Grand Cherokee,2011,14,31626,254706,Fair,9487
Jeep,Grand Cherokee,2012,13,21881,185732,Good,7658
Jeep,Grand Cherokee,2013,12,24869,126396,Fair,9947
Jeep,Grand Cherokee,2014,11,29916,108839,Poor,13462
Jeep,Grand Cherokee,2015,10,33860,56761,Good,16930
Jeep,Grand Cherokee,2016,9,31757,71891,Good,17466
Jeep,Grand Cherokee,2017,8,29579,40409,Good,20413
Jeep,Grand Cherokee,2018,7,29321,80868,Excellent,19058
Jeep,Grand Cherokee,2019,6,19186,118489,Fair,13430
Jeep,Grand Cherokee,2020,5,38135,26209,Excellent,28601
Jeep,Grand Cherokee,2021,4,31909,60796,Poor,25527
Jeep,Grand Cherokee,2022,3,31820,45698,Good,27047
Jeep,Grand Cherokee,2023,2,30674,27899,Excellent,27606
"""

# Step 1: Load and train model
def train_model_from_csv(csv_data: str):
    df = pd.read_csv(StringIO(csv_data))


    X = df.drop("estimated_resale_value", axis=1)
    y = df["estimated_resale_value"]

    categorical_features = ["make", "model", "condition"]
    numeric_features = ["year", "age", "mileage", "original_price"]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), numeric_features)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X, y)
    return pipeline

model = train_model_from_csv(CSV_DATA)

# Step 2: Input validation
def validate_input(make, model, year, age, mileage, condition, original_price):
    if year < 2000 or year > 2024:
        raise ValueError("Year must be between 2000 and 2024")
    if condition not in ["Poor", "Fair", "Good", "Excellent"]:
        raise ValueError("Condition must be one of: Poor, Fair, Good, Excellent")
    if mileage < 0:
        raise ValueError("Mileage cannot be negative")
    if original_price <= 0:
        raise ValueError("Original price must be positive")

def parse_natural_language(prompt: str) -> Dict[str, Union[str, int, float]]:
    """Parse natural language input to extract car details."""
    # Initialize default values
    result = {
        "make": None,
        "model_name": None,
        "year": None,
        "age": None,
        "mileage": None,
        "condition": None,
        "original_price": None
    }
    
    try:
        # Extract year
        year_match = re.search(r'(\d{4})', prompt)
        if year_match:
            result["year"] = int(year_match.group(1))
            result["age"] = 2024 - result["year"]  # Calculate age based on current year
        
        # Extract make and model
        make_model_match = re.search(r'(\w+)\s+(\w+)', prompt)
        if make_model_match:
            result["make"] = make_model_match.group(1)
            result["model_name"] = make_model_match.group(2)
        
        # Extract condition
        condition_match = re.search(r'(poor|fair|good|excellent)', prompt.lower())
        if condition_match:
            result["condition"] = condition_match.group(1).capitalize()
        
        # Extract mileage
        mileage_match = re.search(r'(\d+(?:,\d+)*)\s*mileage', prompt)
        if mileage_match:
            result["mileage"] = float(mileage_match.group(1).replace(',', ''))
        
        # Extract original price
        price_match = re.search(r'(\d+(?:,\d+)*)\s*(?:dollars|dollar|\$)?', prompt)
        if price_match:
            result["original_price"] = float(price_match.group(1).replace(',', ''))
        
        return result
    except Exception as e:
        logger.error(f"Error parsing natural language input: {e}")
        return None

# Step 3: FastMCP tool
@mcp.tool()
async def estimate_resale_value(
    make: str = None,
    model_name: str = None,
    year: int = None,
    age: int = None,
    mileage: float = None,
    condition: str = None,
    original_price: float = None,
    prompt: str = None
) -> str:
    """Estimate the resale value of a car. Can accept either structured parameters or a natural language prompt."""
    try:
        # If prompt is provided, parse it
        if prompt:
            parsed_data = parse_natural_language(prompt)
            if not parsed_data:
                return "Failed to parse the input prompt. Please provide car details in a clear format."
            
            # Update parameters with parsed data
            make = parsed_data["make"]
            model_name = parsed_data["model_name"]
            year = parsed_data["year"]
            age = parsed_data["age"]
            mileage = parsed_data["mileage"]
            condition = parsed_data["condition"]
            original_price = parsed_data["original_price"]
        
        # Validate all required parameters are present
        if not all([make, model_name, year, age, mileage, condition, original_price]):
            return "Missing required parameters. Please provide all car details."
        
        validate_input(make, model_name, year, age, mileage, condition, original_price)
        
        input_df = pd.DataFrame([{
            "make": make,
            "model": model_name,
            "year": year,
            "age": age,
            "mileage": mileage,
            "condition": condition,
            "original_price": original_price
        }])
        
        prediction = model.predict(input_df)[0]
        return f"Estimated resale value: ${round(prediction, 2)}"
    except ValueError as e:
        return f"Input error: {e}"
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return "Failed to estimate resale value due to internal error."

# Step 4: Run the server
if __name__ == "__main__":
    mcp.run(transport="stdio")
