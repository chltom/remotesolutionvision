import pymcprotocol

READ_ADDRS = dict()
WRITE_ADDRS = dict()

PLC1_ID = (0,1,2,3,4,5)
PLC1_IP = "192.168.1.30"
PLC2_ID = (6,7,8,9)
PLC2_IP = "192.168.1.22"

ROBOT_STATUS_BIT_ORDER = (
    "vision_start", 
    "extrun",       # 외부기동
    "extstop",      # 외부정지 
    "reset",  
    "home",         # 대기위치
    "error",        # 로봇이상
    "run",          # 기동중
    "automode", 
    "ready"
)

VISION_STATUS_BIT_ORDER = (
    "vision_start", 
    "vision_complete", 
)

def uint_to_bits(num, bits, small_head=True):
    assert 0 <= num < 2**bits
    binary = bin(int(num))[2:].zfill(bits)
    if small_head:
        binary = reversed(binary)
    return [int(b) for b in binary]


class PLC:
    def __init__(self, ip:str):
        self.__client =  pymcprotocol.Type3E()
        self.__client.setaccessopt(commtype="binary")
        self.__client.connect(ip, 5010)

    def read(self, addrs=[], is_dword=True):
        if isinstance(addrs, int):
            addrs = [addrs]

        addrs = [f"D{a}" for a in addrs if isinstance(a, int)]
        if is_dword:
            _, ret = self.__client.randomread([], addrs)
        else:
            ret, _ = self.__client.randomread(addrs, [])

        if len(ret) == 1:
            ret = ret[0]
        return ret
    
    def write(self, addrs=[], values=[], is_dword=True)->None:
        if isinstance(addrs, int):
            addrs = [addrs]
        if isinstance(values, int):
            values = [values]
        
        assert len(addrs) == len(values)

        addrs = [f"D{a}" for a in addrs if isinstance(a, int)]
        if is_dword:
            self.__client.randomwrite([], [], addrs, values)
        else:
            self.__client.randomwrite(addrs, values, [], [])

    def read_with_key(self, id:int, keys:list[str]):
        if isinstance(keys, str):
            keys = [keys]
        addrs = [READ_ADDRS[id][k] for k in  keys]
        return self.read(addrs)

    def write_with_key(self, id:int, keys:list[str], values:list[int]):
        if isinstance(keys, str):
            keys = [keys]
        if isinstance(values, int):
            values = [values]

        addrs = [WRITE_ADDRS[id][k] for k in keys]
        self.write(addrs, values)

# --------------------------------------------
# id 0, TOP
# --------------------------------------------

READ_ADDRS[0] = dict()

READ_ADDRS[0]["vision_trigger_1"] = 150002
READ_ADDRS[0]["vision_trigger_2"] = 150004
READ_ADDRS[0]["vision_trigger_3"] = 150006

READ_ADDRS[0]["plc_heart_bit"] = 151000
READ_ADDRS[0]["robot_status"] = 151010
READ_ADDRS[0]["plc_model_number"] = 151011

READ_ADDRS[0]["joint1"] = 151020
READ_ADDRS[0]["joint2"] = 151022
READ_ADDRS[0]["joint3"] = 151024
READ_ADDRS[0]["joint4"] = 151026
READ_ADDRS[0]["joint5"] = 151028
READ_ADDRS[0]["joint6"] = 151030

WRITE_ADDRS[0] = dict()
WRITE_ADDRS[0]["vision_heart_bit"] = 151100
WRITE_ADDRS[0]["vision_status"] = 151110

WRITE_ADDRS[0]["x1"] = 151120
WRITE_ADDRS[0]["y1"] = 151122
WRITE_ADDRS[0]["z1"] = 151124
WRITE_ADDRS[0]["rx1"] = 151126
WRITE_ADDRS[0]["ry1"] = 151128
WRITE_ADDRS[0]["rz1"] = 151130

WRITE_ADDRS[0]["x2"] = 151140
WRITE_ADDRS[0]["y2"] = 151142
WRITE_ADDRS[0]["z2"] = 151144
WRITE_ADDRS[0]["rx2"] = 151146
WRITE_ADDRS[0]["ry2"] = 151148
WRITE_ADDRS[0]["rz2"] = 151150

WRITE_ADDRS[0]["x3"] = 151160
WRITE_ADDRS[0]["y3"] = 151162
WRITE_ADDRS[0]["z3"] = 151164
WRITE_ADDRS[0]["rx3"] = 151166
WRITE_ADDRS[0]["ry3"] = 151168
WRITE_ADDRS[0]["rz3"] = 151170

# --------------------------------------------
# id 1, KNOB 1
# --------------------------------------------

READ_ADDRS[1] = dict()
READ_ADDRS[1]["plc_heart_bit"] = 151200
READ_ADDRS[1]["vision_servo_x"] = 151202
READ_ADDRS[1]["vision_servo_y"] = 151204
READ_ADDRS[1]["robot_status"] = 151210
READ_ADDRS[1]["plc_model_number"] = 151211

READ_ADDRS[1]["joint1"] = 151214
READ_ADDRS[1]["joint2"] = 151216
READ_ADDRS[1]["joint3"] = 151218
READ_ADDRS[1]["joint4"] = 151220
READ_ADDRS[1]["joint5"] = 151222
READ_ADDRS[1]["joint6"] = 151224

WRITE_ADDRS[1] = dict()
WRITE_ADDRS[1]["vision_heart_bit"] = 151300
WRITE_ADDRS[1]["vision_status"] = 151310
WRITE_ADDRS[1]["vision_model_number"] = 151311

WRITE_ADDRS[1]["x"] = 151314
WRITE_ADDRS[1]["y"] = 151316
WRITE_ADDRS[1]["z"] = 151318
WRITE_ADDRS[1]["rx"] = 151320
WRITE_ADDRS[1]["ry"] = 151322
WRITE_ADDRS[1]["rz"] = 151324

WRITE_ADDRS[1]["x1"] = 151328
WRITE_ADDRS[1]["y1"] = 151330
WRITE_ADDRS[1]["z1"] = 151332
WRITE_ADDRS[1]["rx1"] = 151334
WRITE_ADDRS[1]["ry1"] = 151336
WRITE_ADDRS[1]["rz1"] = 151338

# --------------------------------------------
# id 2, KNOB 2
# --------------------------------------------

READ_ADDRS[2] = dict()
READ_ADDRS[2]["plc_heart_bit"] = 151200
READ_ADDRS[2]["vision_servo_x"] = 151206
READ_ADDRS[2]["vision_servo_y"] = 151208

READ_ADDRS[2]["robot_status"] = 151240
READ_ADDRS[2]["plc_model_number"] = 151241

READ_ADDRS[2]["joint1"] = 151244
READ_ADDRS[2]["joint2"] = 151246
READ_ADDRS[2]["joint3"] = 151248
READ_ADDRS[2]["joint4"] = 151250
READ_ADDRS[2]["joint5"] = 151252
READ_ADDRS[2]["joint6"] = 151254

WRITE_ADDRS[2] = dict()
WRITE_ADDRS[2]["vision_heart_bit"] = 151300
WRITE_ADDRS[2]["vision_status"] = 151340
WRITE_ADDRS[2]["vision_model_number"] = 151341

WRITE_ADDRS[2]["x"] = 151344
WRITE_ADDRS[2]["y"] = 151346
WRITE_ADDRS[2]["z"] = 151348
WRITE_ADDRS[2]["rx"] = 151350
WRITE_ADDRS[2]["ry"] = 151352
WRITE_ADDRS[2]["rz"] = 151354

# --------------------------------------------
# id 3, Rubber
# --------------------------------------------

READ_ADDRS[3] = dict()
READ_ADDRS[3]["plc_heart_bit"] = 151200
READ_ADDRS[3]["robot_status"] = 151270
READ_ADDRS[3]["plc_model_number"] = 151271

READ_ADDRS[3]["joint1"] = 151274
READ_ADDRS[3]["joint2"] = 151276
READ_ADDRS[3]["joint3"] = 151278
READ_ADDRS[3]["joint4"] = 151280
READ_ADDRS[3]["joint5"] = 151282
READ_ADDRS[3]["joint6"] = 151284

WRITE_ADDRS[3] = dict()
WRITE_ADDRS[3]["vision_heart_bit"] = 151300
WRITE_ADDRS[3]["vision_status"] = 151370
WRITE_ADDRS[3]["vision_model_number"] = 151371

WRITE_ADDRS[3]["x"] = 151374
WRITE_ADDRS[3]["y"] = 151376
WRITE_ADDRS[3]["z"] = 151378
WRITE_ADDRS[3]["rx"] = 151380
WRITE_ADDRS[3]["ry"] = 151382
WRITE_ADDRS[3]["rz"] = 151384

# --------------------------------------------
# id 4, PCB
# --------------------------------------------

READ_ADDRS[4] = dict()
READ_ADDRS[4]["plc_heart_bit"] = 151400
READ_ADDRS[4]["robot_status"] = 151410
READ_ADDRS[3]["plc_model_number"] = 151411

READ_ADDRS[4]["joint1"] = 151420
READ_ADDRS[4]["joint2"] = 151422
READ_ADDRS[4]["joint3"] = 151424
READ_ADDRS[4]["joint4"] = 151426
READ_ADDRS[4]["joint5"] = 151428
READ_ADDRS[4]["joint6"] = 151430

WRITE_ADDRS[4] = dict()
WRITE_ADDRS[4]["vision_heart_bit"] = 151500
WRITE_ADDRS[4]["vision_status"] = 151510
WRITE_ADDRS[4]["vision_model_number"] = 151511

WRITE_ADDRS[4]["x"] = 151520
WRITE_ADDRS[4]["y"] = 151522
WRITE_ADDRS[4]["z"] = 151524
WRITE_ADDRS[4]["rx"] = 151526
WRITE_ADDRS[4]["ry"] = 151528
WRITE_ADDRS[4]["rz"] = 151530

# --------------------------------------------
# id 5, BOTTOM
# --------------------------------------------

READ_ADDRS[5] = dict()
READ_ADDRS[5]["plc_heart_bit"] = 151600
READ_ADDRS[5]["robot_status"] = 151610
READ_ADDRS[5]["plc_model_number"] = 151611

READ_ADDRS[5]["joint1"] = 151640
READ_ADDRS[5]["joint2"] = 151642
READ_ADDRS[5]["joint3"] = 151644
READ_ADDRS[5]["joint4"] = 151646
READ_ADDRS[5]["joint5"] = 151648
READ_ADDRS[5]["joint6"] = 151650

WRITE_ADDRS[5] = dict()
WRITE_ADDRS[5]["vision_heart_bit"] = 151700
WRITE_ADDRS[5]["vision_status"] = 151710
WRITE_ADDRS[5]["vision_model_number"] = 151711

WRITE_ADDRS[5]["x"] = 151740
WRITE_ADDRS[5]["y"] = 151742
WRITE_ADDRS[5]["z"] = 151744
WRITE_ADDRS[5]["rx"] = 151746
WRITE_ADDRS[5]["ry"] = 151748
WRITE_ADDRS[5]["rz"] = 151750

# --------------------------------------------
# id 6, FUNCTION
# --------------------------------------------

READ_ADDRS[6] = dict()
READ_ADDRS[6]["plc_heart_bit"] = 151000
READ_ADDRS[6]["robot_status"] = 151010
READ_ADDRS[6]["plc_model_number"] = 151011

READ_ADDRS[6]["joint1"] = 151040
READ_ADDRS[6]["joint2"] = 151042
READ_ADDRS[6]["joint3"] = 151044
READ_ADDRS[6]["joint4"] = 151046
READ_ADDRS[6]["joint5"] = 151048
READ_ADDRS[6]["joint6"] = 151050

WRITE_ADDRS[6] = dict()
WRITE_ADDRS[6]["vision_heart_bit"] = 151100
WRITE_ADDRS[6]["vision_status"] = 151110
WRITE_ADDRS[6]["vision_model_number"] = 151111

WRITE_ADDRS[6]["x"] = 151140
WRITE_ADDRS[6]["y"] = 151142
WRITE_ADDRS[6]["z"] = 151144
WRITE_ADDRS[6]["rx"] = 151146
WRITE_ADDRS[6]["ry"] = 151148
WRITE_ADDRS[6]["rz"] = 151150

# --------------------------------------------
# id 7, Inspection1
# --------------------------------------------

READ_ADDRS[7] = dict()
READ_ADDRS[7]["plc_heart_bit"] = 151200
READ_ADDRS[7]["robot_status"] = 151210
READ_ADDRS[7]["plc_model_number"] = 151211

READ_ADDRS[7]["joint1"] = 151240
READ_ADDRS[7]["joint2"] = 151242
READ_ADDRS[7]["joint3"] = 151244
READ_ADDRS[7]["joint4"] = 151246
READ_ADDRS[7]["joint5"] = 151248
READ_ADDRS[7]["joint6"] = 151250

WRITE_ADDRS[7] = dict()
WRITE_ADDRS[7]["vision_heart_bit"] = 151300
WRITE_ADDRS[7]["vision_status"] = 151310
WRITE_ADDRS[7]["vision_model_number"] = 151311

WRITE_ADDRS[7]["x"] = 151340
WRITE_ADDRS[7]["y"] = 151342
WRITE_ADDRS[7]["z"] = 151344
WRITE_ADDRS[7]["rx"] = 151346
WRITE_ADDRS[7]["ry"] = 151348
WRITE_ADDRS[7]["rz"] = 151350


# --------------------------------------------
# id 8, Inspection2
# --------------------------------------------

READ_ADDRS[8] = dict()
READ_ADDRS[8]["plc_heart_bit"] = 151400
READ_ADDRS[8]["robot_status"] = 151410
READ_ADDRS[8]["plc_model_number"] = 151411

READ_ADDRS[8]["joint1"] = 151440
READ_ADDRS[8]["joint2"] = 151442
READ_ADDRS[8]["joint3"] = 151444
READ_ADDRS[8]["joint4"] = 151446
READ_ADDRS[8]["joint5"] = 151448
READ_ADDRS[8]["joint6"] = 151450

WRITE_ADDRS[8] = dict()
WRITE_ADDRS[8]["vision_heart_bit"] = 151500
WRITE_ADDRS[8]["vision_status"] = 151510
WRITE_ADDRS[8]["vision_model_number"] = 151511

WRITE_ADDRS[8]["x"] = 151540
WRITE_ADDRS[8]["y"] = 151542
WRITE_ADDRS[8]["z"] = 151544
WRITE_ADDRS[8]["rx"] = 151546
WRITE_ADDRS[8]["ry"] = 151548
WRITE_ADDRS[8]["rz"] = 151550

# --------------------------------------------
# id 9, Battery Cover
# --------------------------------------------

READ_ADDRS[9] = dict()
READ_ADDRS[9]["plc_heart_bit"] = 151600
READ_ADDRS[9]["robot_status"] = 151610
READ_ADDRS[9]["plc_model_number"] = 151611

READ_ADDRS[9]["joint1"] = 151640
READ_ADDRS[9]["joint2"] = 151642
READ_ADDRS[9]["joint3"] = 151644
READ_ADDRS[9]["joint4"] = 151646
READ_ADDRS[9]["joint5"] = 151648
READ_ADDRS[9]["joint6"] = 151650

WRITE_ADDRS[9] = dict()
WRITE_ADDRS[9]["vision_heart_bit"] = 151700
WRITE_ADDRS[9]["vision_status"] = 151710
WRITE_ADDRS[9]["vision_model_number"] = 151711

WRITE_ADDRS[9]["x"] = 151740
WRITE_ADDRS[9]["y"] = 151742
WRITE_ADDRS[9]["z"] = 151744
WRITE_ADDRS[9]["rx"] = 151746
WRITE_ADDRS[9]["ry"] = 151748
WRITE_ADDRS[9]["rz"] = 151750
