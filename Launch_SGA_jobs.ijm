#@String (label="Username", value="username", description="Username on the host") username
#@String (label="Host", value="hex", description="hostname to which command are send") hostname
#@File (label="Local share",description="Local mounting point of the network share", style="directory") local_share
#@String (label="Remote share",description="Remote mounting point of the network share", value="/cephfs/") remote_share
#@File(label="Folder", value="", style="directory",description="Path to the data folder from this computer") folder
#@String(label="Action",choices={"Install","Scan","Config","Process","Figure","List Jobs"}) action

/*
 * Launch slurm jobs for stress granule analysis
 *
 * Workflow :
 * 1. Install the script
 * 2. Scan the folder where the files are and create a filelist.csv saved in that folder
 * 3. Process the file list, open the table and create jobs for each file to run on the cluster
 * 4. Make a figure with all the individual results.
 *
 * Jerome Boulanger 2021-22
 */

print("\n\n__________SGA______________");

remote_path = replace(convert_slash(folder), convert_slash(local_share), remote_share);
remote_jobs_dir = remote_share + "/jobs";
local_jobs_dir = local_share + File.separator + "jobs";
script_url = "https://raw.githubusercontent.com/jboulanger/sganalysis/master/sganalysiswf.py";
script_name = "sganalysiswf.py";

// create a job folder if needed
if (File.exists(local_jobs_dir) != 1) {
	print("Creating a jobs folder in " + local_share);
	File.makeDirectory(local_jobs_dir);
} else {
	print("Jobs folder already present in " + local_share);
}

if (matches(action, "Scan")) {
	scan();
} else if (matches(action, "Config")) {
	config();
} else if (matches(action, "Process")) {
	process();
}else if (matches(action, "Figure")) {
	figure();
} else if (matches(action, "Install"))  {
	install();
} else if (matches(action, "List Jobs"))  {
	listjobs();
}

function install() {
	print("[ Installing script ]");
	print("Downloading the python script and save it in the job folder");
	str = File.openUrlAsString(script_url);
	dst = local_jobs_dir + File.separator + "sganalysiswf.py";
	File.saveString(str,local_jobs_dir+File.separator+script_name);
	print("Done");
}

function scan() {
	print("[ Scanning data folder ]");
	print("List all files in the data folder and create a filelist.csv file.");
	print(" - Remote path " + remote_path);
	print(" - File list " + remote_path+"/filelist.csv");
	jobname = "sga-scan.sh";
	str  = "#!/bin/tcsh\n#SBATCH --job-name=sg-scan\n#SBATCH --time=01:00:00\nconda activate sganalysis\npython sganalysiswf.py scan --data-path=\""+remote_path+"\" --file-list \""+remote_path+"/filelist.csv\" --config \""+remote_path+"/config.json\"";	
	File.saveString(str,local_jobs_dir+File.separator+jobname);
	ret = exec("ssh", username+"@"+hostname, "sbatch", "--chdir", remote_jobs_dir, jobname);
	print(" " + ret);
	print(" Job is now running in the background, use 'List Jobs' to check completion.");
	print(" Next: edit channel orders in the config.json file.");	
}

function config() {
	print("[ Configuration file ]");
	Dialog.create("Channel Configuration Tool");
	Dialog.addString("Channels labels", "other,membrane,granule,nuclei");
	Dialog.addString("Channels wavelength", "525,620,700,447");
	Dialog.show();
	channels_str = Dialog.getString();
	waves_str = Dialog.getString();
	channels = parseCSVString(channels_str);
	waves = parseCSVString(waves_str);
	str = "{\"channels\":[";
	for (i = 0; i < channels.length; i++) {
		str += "{\"index\":"+i+",\"name\":\""+channels[i] +"\", \"wavelength\":"+waves[i]+"}";
		if (i!= channels.length-1) {
			str+=",";
			}
	}
	str += "],\"NA\":0.95,\"medium_refractive_index\":1.4, \"scale_um\":50}";
	
	print("Saving configuration file in folder");
	print(folder+File.separator+"config.json");
	File.saveString(str, folder+File.separator+"config.json");
	print("Configuration file has been created, ready to process the dataset.");
}

function process() {
	print("[ Processing a list of files ]");
	print("Open filelist.csv and create a job for each time.");
	if (!File.exists(folder+File.separator+"results")) {
		print("Creating results directory");
		File.makeDirectory(folder+File.separator+"results");
	}
	jobname = "sga-process.sh";
	str  = "#!/bin/tcsh\n#SBATCH --job-name=sga-process\n#SBATCH --time=05:00:00\n#SBATCH --partition=gpu\n#SBATCH --gres=gpu:1\nconda activate sganalysis\npython sganalysiswf.py process --data-path=\""+remote_path+"\" --file-list \""+remote_path+"/filelist.csv\" --index $SLURM_ARRAY_TASK_ID --output-by-cells \""+remote_path+"\"/results/cells-$SLURM_ARRAY_TASK_ID.csv --output-vignette \""+remote_path+"\"/results/vignettes-$SLURM_ARRAY_TASK_ID.png";
	File.saveString(str,local_jobs_dir+File.separator+jobname);
	Table.open(folder+File.separator+"filelist.csv");
	n = Table.size;
	ret = exec("ssh", username+"@"+hostname, "sbatch", "--chdir", remote_jobs_dir,  "--array=1-"+n, jobname);
	print(ret);
	print("Job is running in the background, use 'List Jobs' to check completion.");
	print("Next: create figures using the 'Figure' action.");	
}

function figure() {
	print("[ Preparing a figure ]");	
	jobname = "sga-fig.sh";
	str  = "#!/bin/tcsh\n#SBATCH --job-name=sg-fig\n#SBATCH --time=01:00:00\nconda activate sganalysis\npython sganalysiswf.py figure --data-path=\""+remote_path+"\" --file-list \""+remote_path+"/filelist.csv\"";
	File.saveString(str,local_jobs_dir+File.separator+jobname);
	ret = exec("ssh", username+"@"+hostname, "sbatch", "--chdir", remote_jobs_dir, jobname);
	print(ret);
}

function listjobs() {
	print("[ List jobs ]");	
	if (isOpen("Results")) { selectWindow("Results"); run("Close"); }
	ret = exec("ssh", username+"@"+hostname, "squeue -u "+username);
	lines = split(ret,"\n");
	if (lines.length == 1) {
		print("No jobs are running");
	}

	for (i = 1; i < lines.length; i++) {
		elem = split(lines[i]," ");
		setResult("JOBID",i-1,elem[0]);
		setResult("PARTITION",i-1,elem[1]);
		setResult("NAME",i-1,elem[2]);
		setResult("USER",i-1,elem[3]);
		setResult("STATUS",i-1,elem[4]);
		setResult("TIME",i-1,elem[5]);
	}
}

function convert_slash(src) {
	// convert windows file separator to unix file separator if needed
	if (File.separator != "/") {
		a = split(src,File.separator);
		dst = "";
		for (i=0;i<a.length-1;i++) {
			dst += a[i] + "/";
		}
		dst += a[a.length-1];
		return dst;
	} else {
		return src;
	}
}

function parseCSVString(csv) {	
	str = split(csv, ",");
	for (i = 0; i < str.length; i++) { 
		str[i] =  String.trim(str[i]);
	}
	return str;
}