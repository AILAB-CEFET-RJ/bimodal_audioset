import csv, sys
import os
import wave
import contextlib
import argparse



project_path = os.path.dirname(os.path.abspath(__file__))
balanced_train_segments = project_path+'/segments/subset_balanced_train_segments.csv'
unbalanced_train_segments = project_path+'/segments/2000max_subset_unbalanced_train_segments.csv'
eval_segments = project_path+'/segments/subset_eval_segments.csv'
data_path = project_path+'/data/'


# specify the index of files that is downloaded last time (to resume downloading)
# Basically this is a simple work around for this downloader, where I sometimes accidentally close the program or sometimes it just hangs in my environment
last_processed_row = 0

def youtube_downloader(data_path ,subfolder, id, start_time, idx):
    ret = os.system('ffmpeg -n $(youtube-dl --get-url \'https://www.youtube.com/watch?v=' + id + '\' '
      '-f \'(mp4)[height<480]\' | sed "s/.*/-ss '+start_time+' -i &/") '
      '-t 00:10 -c copy ' + data_path + subfolder +'video/'+ idx + '_' + id + '.mp4')
    ## save audio
    if ret == 0:
      os.system('ffmpeg -i '+ data_path + subfolder +'video/'+ idx + '_' + id + '.mp4 -vn -acodec copy '+ data_path + subfolder +'audio/'+ idx + '_' + id + '.aac')
    return ret


def create_error_file(path,id, idx):
    with open(path + "errors.txt", "a") as myfile:
        myfile.write(idx + '_' + id +"\n")


def download_data(segments,subfolder):

    rownum = 0

    if not os.path.exists(data_path+subfolder+'video/'):
        os.makedirs(data_path+subfolder+'video/')
    if not os.path.exists(data_path+subfolder+'audio/'):
        os.makedirs(data_path+subfolder+'audio/')


    with open(segments, newline='') as f:
        reader = csv.reader(f)
        try:
            for row in reader:
                if rownum <= last_processed_row:
                  rownum += 1
                  continue
                # Skip the 3 line header
                if rownum >= 3:
                    print(row)

                    if (os.path.exists(data_path + subfolder +'video/'+ str(rownum - 3) + '_' + row[0] + '.mp4')):
                        print("file exists, skipping...")
                        rownum += 1
                        continue

                    ret = youtube_downloader(data_path,subfolder,row[0], str(float(row[1].lstrip())),str(rownum - 3))
                    # If there was an error downloading the file
                    # This sometimes happens if videos are blocked or taken down
                    if ret != 0:
                        create_error_file(data_path+subfolder,row[0], str(rownum - 3))

                rownum += 1

        except csv.Error as e:
            sys.exit('file {}, line {}: {}'.format(segments, reader.line_num, e))


if __name__ == '__main__':


    parser = argparse.ArgumentParser(
        description='Directy download from youtube the videos and audio files of youtube audioset.')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval',  action='store_true')
    parser.add_argument('--unbalanced_train', action='store_true')


    if len(sys.argv) < 2:
        parser.print_usage()
        sys.exit(1)

    args = parser.parse_args()


    # Only use what you need
    if args.train:
        print('Downloading balanced trainig datased defined in',balanced_train_segments)
        download_data(balanced_train_segments,"balanced_train/")
    if args.eval:
        print('Downloading evaluation datased defined in',eval_segments)
        download_data(eval_segments,"eval/")
    if args.unbalanced_train:
        print('Downloading unbalanced training datased defined in',unbalanced_train_segments)
        download_data(unbalanced_train_segments,"2000unbalanced_train/")
