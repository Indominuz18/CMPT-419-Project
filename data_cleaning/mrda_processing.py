import xml.etree.ElementTree as ET
import os
import glob
from pydub import AudioSegment

processed_audio_folder = "split_audiov2"
audio_folder = "audio"
padding = 100 # milliseconds of padding between start and end

dialogue_act_path = './data_cleaning/' + '/dialogue-acts/'
files = [os.path.basename(x) for x in glob.glob(dialogue_act_path + "*.xml")]

# for every annotated audio file, split up the single audio file into multiple audio files containing
# the declarative question
for file in files:
    split_file_name = file.split(".") 
    folder_name = split_file_name[0]
    partition = split_file_name[1]
    print(folder_name)
    print(partition)

    audio_file_path = "./data_cleaning/" + processed_audio_folder + "/" + folder_name + "/"

    # don't process already processed audio clips
    # if os.path.exists(audio_file_path):
    #     continue

    da_tree = ET.parse(dialogue_act_path + folder_name + '.' + partition + '.dialogue-acts.xml')
    root = da_tree.getroot()

    # get the audio to split
    interactionAudio = AudioSegment.from_wav("./data_cleaning/" + audio_folder + "/" + folder_name + "/" + folder_name + ".interaction.wav")

    for i, child in enumerate(root):
        # determine if there is a labelled declarative question in the xml files
        if 'type' in child.attrib:
            dialogue_type = child.attrib['type']
        else:
            continue

        # checking the type of question
        if "qy^d" in dialogue_type:
            # audio splitting works on milliseconds
            start_time = float(child.attrib["starttime"]) * 1000
            end_time = float(child.attrib["endtime"]) * 1000

            # remove all short audio clips less than a certain threshold
            if end_time - start_time < 250:
                continue

            start_time -= padding
            end_time += padding

            interactionAudio_split = interactionAudio[start_time:end_time]
            
            # create the file path if it doesn't already exist
            if not os.path.exists(audio_file_path):
                os.makedirs(audio_file_path)
                
            interactionAudio_split.export(audio_file_path + partition + str(i) + '_audio.wav', format="wav")