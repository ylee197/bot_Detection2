# Bot_Detecter


## Introduction
This code detect twitter bots. This algorithm is inspired by Coornet algorithm. 
Coornet algorithm detectes communities that presenting unuseal behaviors.
What is differences betweetn CooRnet and this program?
- CooRnet detects the community. This program detects suspicious user_ids.
- This program is faster than CooRnet.
- This program can process larget dataset than CooRnet. 

## Directory Structure
- code - Bot detecting program
- data - Images
## Execution

### Environment
- Python 3.8.8
- numpy 1.20.1
- pandas 1.2.4
- matplotlib 3.3.4

### Execution instruction
1. Downloading 'code' folder
2. Preparing input file. You need to follow the 'Input file' instruction that is below. 
3. Please locate your input file under code folder. You should put your input file in the same folder with run.py file. In that way, you don't need to designate the --data_path when you run the program. 
4. You should put step that you want to excute it. There are five steps.<br/> 
   : It is better to execute following the order step1 - step2 - step3 - step3(Adjusted BIN_SIZE) - step4 - step4(Adjusted BIN_SIZE) - step5.<br/> You could run step3 and step4 multipul times to find proper BIN_SIZE. 
   - step1 : Data preprocessing
   - step2 : Setting the input file format and making time interval histogram. 
   - step3 : Creating Coordinated histogram of fast bots. 
   - step4 : Creating Coordinated histogram of slow bots. 
   - step5 : Making a bot list. 

5. Running the program with below command:
   - step1 : python3 ./run.py --Step 1 --file_name INPUT_FILE_NAME
   - step2 : python3 ./run.py --Step 2 --file_name INPUT_FILE_NAME --Time_threshold TIME_THRESHOLD 
   - step3 : python3 ./run.py --Step 3 --file_name INPUT_FILE_NAME --bin BIN_SIZE
   - step4 : python3 ./run.py --Step 4 --file_name INPUT_FILE_NAME --bin BIN_SIZE --fast_threshold FAST_THRESHOLD
   - step5 : python3 ./run.py --Step 5 --file_name INPUT_FILE_NAME --fast_threshold FAST_THRESHOLD --slow_threshold SLOW_THRESHOLD 

### Setting threshold
: You need to set thresholds.
1. <B> TIME_THRESHOLD </B> : After investigating a 'repost_interval_ID.png' histogram, you need to set a time_threshold. Then, you need to add TIME_THRESHOLD to the step2 command line.<br/>
   For example, TIME_THRESHOLD is 11sec in the below histogram. <br/>
   In this case step 2 command line should be <B>'python3 ./run.py --step 2 --file_name INPUT_FILE_NAME --time_threshold 11'</B><br/>
   ![alt text](https://github.com/ylee197/bot_Detection/blob/main/data/Time_interval.png?raw=true)
2. <B> FAST_THRESHOLD and SLOW_THRESHOLD</B> uses coordinated reporting histogram. Coordinated reposting histogram will be a gradually decreased long tail histogram. However, if there are bot activities, the gradually decreased long tail histogram lose its regular pattern. Thus, if there is a irregular pattern in the histogram such as not decreased pattern and increased pattern that point will be a threshold. <br/>
   In the below graph, the histogram is decrased less after 6. Thus, the threshold of this graph is 6. This rule is aplied to both FAST_THRESHOLD and SLOW_THRESHOLD.<br />
   ![alt text](https://github.com/ylee197/bot_Detection/blob/main/data/Coordinated.png?raw=true)
3. <B> BIN_SIZE </B> : After investigating a histogram_fast.png(small/small_log) and a histogram_slow.png(small/small_log) histograms, yon can set a BIN_SIZE. If you find gaps in a histogram_fast_small.png or a histogram_fast_small_log.png, you need to reduce yout BIN_SIZE. At the first step3 running, program is running with default BIN_SIZE(default BIN_SIZE is 50). 
   ![alt text](https://github.com/ylee197/bot_Detection/blob/main/data/Gap.png?raw=true)
5. <B> FAST_THRESHOLD </B> : After investigating a histogram_fast.png(small/small_log) histogram, you need to set a fast_threshold. Then, you need to add FAST_THRESHOLD to the step4 command line.<br/> For example, if FAST_THRESHOLD is 10, step4 command line is <B>python3 ./run.py --step 4 --file_name INPUT_FILE_NAME --bin BIN_SIZE --fast_threshold 10</B>
6. <B> SLOW_THRESHOLD </B> : After investigating a histogram_slow.png(small/small_log) histogram, you need to set a slow_threshold. Then, you need to add FAST_THRESHOLD and SLOW_THRESHOLD to the step5 command line.<br/> For example, if FAST_THRESHOLD is 10 and SLOW_THRESHOLD is 8, step5 command line is <B>python3 ./run.py --step 5 --file_name INPUT_FILE_NAME --fast_threshold 10 --slow_threshold 8 </B>
  
## Input file
- Input file with extention .csv
- Input file should include below columns:
  - <B>tid</B> : Tweet id
  - <B>user</B> : A user identifier. This user is an author of a post.
  - <B>target_user</B> : A user identifier. This user is an author of a being reposted post. 
  - <B>target</B> : Being retweeted target identifier. This can be either 'unwounded_url' or 'retweeted_id'
  - <B>timestamp</B> : Time that twitter is created. This data format should be "YYYY-MM-DD HH:MM:SS."
  - Tid and target should be the same format. For the format, you can use either string or id numbers. 
  - User and target_user should be the same format. For the format, it should be the tid format. 
- Input file columns should follow this order: |'tid','user','target_user','target','timestamp'|

## Output files
Output file is saved under "data" folder
** Histogram files such as repost_interval_ID.png, histogram_fast.png and histogram_slow.png have three versions - full, small, small_log. In many cases, it is hard to be recognised the irregular pattern in the full histogram files. 
   - <B>full</B> : Full histogram. 
   - <B>small</B> : Showing the beginning part of the histogram. To set up the beginning part of the histogram you what to check, you need to put the bin size at small_histogram_bin_size in "bot_detector.py." 
   - <B>small_log</B> : In many cases, there are big differences between the first few bins and last of it. It makes hard to recognising the irregular pattern. Log graph make it easy to check the changes. If you have a hard time to find the threshold, I recomend to use this small_log graph.     
- File list
  - <B>repost_interval_ID.png(full/small)</B> : A time interval histogram between previous retweet and next retweet.
  - <B>histogram_fast.png(small/small_log)</B> : A histogram presents the volumne of coordinated retweets of fast IDs. 
  - <B>histogram_slow.png(small/small_log)</B> : A histogram presents the volumne of coordinated retweets of slow IDs. 
  - <B>report.txt</B> : a total report of bot detecting process
  - <B>suspicious_ID_list.csv</B> : A list of twitter IDs. These IDs have short period of retweet period and shows coordinated behavior many times. Thus, thouse Ids are highly suspicious IDs. 
 

## Created by
Yeonjung Lee
