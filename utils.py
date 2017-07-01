import  os
import numpy
import cv2
import wave, pyaudio

def read_faces(root_path):
    X, y=[], []

    for dirName, dirNames, fileNames in os.walk(root_path):
        for subDirName in dirNames:
            subDirFullPath=os.path.join(dirName, subDirName)
            y_type=gety(subDirName)
            for fileName in os.listdir(subDirFullPath):
                fullFilePath=os.path.join(subDirFullPath, fileName)
                faceImage=cv2.imread(fullFilePath, cv2.IMREAD_GRAYSCALE)
                X.append(numpy.asarray(faceImage, dtype=numpy.uint8))
                y.append(y_type)

    return [X, y]

def gety(dirName):
    if(dirName=="0"):
        return 0
    elif(dirName=="1"):
        return 1
    else:
        return 2


def getName(y):
    if(y==0):
        return "dai bin hua"
    elif(y==1):
        return "dai yi xuan"
    else:
        return "UNKNOW"


def play_sound(wavFile):
    # define stream chunk
    chunk = 1024

    # open a wav format music
    f = wave.open(r'C:\\Users\\Administrator\\PycharmProjects\\face_detector\\sounds\\'+wavFile, "rb")
    # instantiate PyAudio
    p = pyaudio.PyAudio()
    # open stream
    stream = p.open(format=p.get_format_from_width(f.getsampwidth()),
                    channels=f.getnchannels(),
                    rate=f.getframerate(),
                    output=True)
    # read data
    data = f.readframes(chunk)

    # paly stream
    while data != b'':
        stream.write(data)
        data = f.readframes(chunk)

    # stop stream
    stream.stop_stream()
    stream.close()

    # close PyAudio
    p.terminate()

    f.close()
