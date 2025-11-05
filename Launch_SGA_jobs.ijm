#@String (label="Username", value="username", description="Username on the host") username
#@String (label="Host", value="hex", description="hostname to which command are send") hostname
#@File (label="Local share",description="Local mounting point of the network share", style="directory") local_share
#@String (label="Remote share",description="Remote mounting point of the network share", value="/cephfs/") remote_share
#@File(label="Folder", value="", style="directory",description="Path to the data folder from this computer") folder
#@String(label="Python", value="", choices={"conda","micromamba","common"}, description="Type of python installation") python
#@String(label="Action",choices={"Install","Scan ND2","Scan LSM","Scan TIFF","Config SG","Config Spread","Process","Figure","List Jobs","Cancel Jobs","Open first image"}) action
#@Boolean(label="GPU queue",value=True) use_gpu_queue

/*
 * Launch slurm jobs for stress granule analysis
 *
 * Workflow :
 * 1. Install the script
 * 2. Scan the folder where the files are and create a filelist.csv saved in that folder
 * 3. Process the file list, open the table and create jobs for each file to run on the cluster
 * 4. Make a figure with all the individual results.
 *
 * Jerome Boulanger 2021-25
 */

print("\n\n__________SGA______________");
getDateAndTime(year, month, dayOfWeek, dayOfMonth, hour, minute, second, msec);
print(""+year+"/"+month+"/"+dayOfMonth+" "+hour+":"+minute+":"+second);
remote_path = replace(convert_slash(folder), convert_slash(local_share), remote_share);
remote_jobs_dir = remote_share + "/jobs";
local_jobs_dir = local_share + File.separator + "jobs";
script_url = "https://raw.githubusercontent.com/jboulanger/sganalysis/master/sganalysiswf.py";
env_url = "https://raw.githubusercontent.com/jboulanger/sganalysis/master/environment.yml";
script_name = "sganalysiswf.py";
if (python == "conda") {
	cmd = "~/miniconda3/bin/conda run -n sganalysis python " + script_name  + " ";
} else if (python == "micromamba"){
	cmd = "~/.local/bin/micromamba run -n sganalysis python "  + script_name  + " ";
} else {
	cmd = "~/.local/bin/micromamba -p /lmb/home/jeromeb/micromamba/envs/sganalysis run python " + script_name  + " ";
}


// create a job folder if needed
if (File.exists(local_jobs_dir) != 1) {
	print("Creating a jobs folder in " + local_share);
	File.makeDirectory(local_jobs_dir);
}

if (matches(action, "Scan ND2")) {
	scannd2();
} else if(matches(action, "Scan LSM")){
	scanlsm();
} else if(matches(action, "Scan TIFF")){
	scantiff();
} else if (matches(action, "Config SG")) {
	configSG();
} else if (matches(action, "Config Spread")) {
	configSpread();
} else if (matches(action, "Process")) {
	process();
}else if (matches(action, "Figure")) {
	figure();
} else if (matches(action, "Install")) {
	install();
} else if (matches(action, "List Jobs")) {
	listJobs();
} else if (matches(action, "Cancel Jobs")) {
	cancelJobs();
} else if (matches(action, "Open first image")) {
	Table.open(folder+File.separator+"filelist.csv");
	fname = Table.getString("filename", 0);
	print("Opening image" + fname);
	open(folder + File.separator + fname);
}

function install() {
	print("[ Installing script in your job folder ]");
	print("Downloading the python script and save it in the job folder");
	str = File.openUrlAsString(script_url);
	dst = local_jobs_dir + File.separator + script_name;
	File.saveString(str,local_jobs_dir+File.separator+script_name);
	print("Done");
}

function scannd2() {
	print("[ Scanning data folder for ND2 files]");
	print("List all files in the data folder and create a filelist.csv file.");
	print(" - Remote path " + remote_path);
	print(" - File list " + remote_path+"/filelist.csv");
	jobname = "sga-scan.sh";
	str  = "#!/bin/bash\n";
	str += "#SBATCH --job-name=sg-scan\n";
	str += "#SBATCH --time=01:00:00\n";
	str += cmd + "scan --file-type nd2 --data-path=\""+remote_path+"\" --file-list \""+remote_path+"/filelist.csv\" --config \""+remote_path+"/config.json\"";
	File.saveString(str,local_jobs_dir+File.separator+jobname);
	ret = exec("ssh", username+"@"+hostname, "sbatch", "--chdir", remote_jobs_dir, jobname);
	print(" " + ret);
	print(" Jobs are now running in the background, use 'List Jobs' to check completion.");
	print(" Next: edit channel orders in the config.json file.");
}

function scanlsm() {
	print("[ Scanning data folder for LSM files ]");
	print("List all files in the data folder and create a filelist.csv file.");
	print(" - Remote path " + remote_path);
	print(" - File list " + remote_path+"/filelist.csv");
	jobname = "sga-scan.sh";
	str  = "#!/bin/bash\n";
	str += "#SBATCH --job-name=sg-scan\n";
	str += "#SBATCH --time=01:00:00\n";
	str += "pwd\n";
	str += cmd + "scan --file-type lsm --data-path=\""+remote_path+"\" --file-list \""+remote_path+"/filelist.csv\" --config \""+remote_path+"/config.json\"";
	File.saveString(str, local_jobs_dir + File.separator + jobname);
	ret = exec("ssh", username+"@"+hostname, "sbatch", "--chdir", remote_jobs_dir, jobname);
	print(" " + ret);
	print(" Job are now running in the background, use 'List Jobs' to check completion.");
	print(" Next: edit channel orders in the config.json file.");
}

function scantiff() {
	print("[ Scanning data folder for TIFF files ]");
	print("List all files in the data folder and create a filelist.csv file.");
	print(" - Remote path " + remote_path);
	print(" - File list " + remote_path+"/filelist.csv");
	jobname = "sga-scan.sh";
	str  = "#!/bin/bash\n";
	str += "#SBATCH --job-name=sg-scan\n";
	str += "#SBATCH --time=01:00:00\n";
	str += "pwd\n";
	str += cmd + "scan --file-type tiff --data-path=\""+remote_path+"\" --file-list \""+remote_path+"/filelist.csv\" --config \""+remote_path+"/config.json\"";
	File.saveString(str, local_jobs_dir + File.separator + jobname);
	ret = exec("ssh", username+"@"+hostname, "sbatch", "--chdir", remote_jobs_dir, jobname);
	print(" " + ret);
	print(" Job are now running in the background, use 'List Jobs' to check completion.");
	print(" Next: edit channel orders in the config.json file.");
}

function configSG() {
	print("[ Configuration file for stress granule analysis]");
	print(folder);
	Table.open(folder+"/filelist.csv");
	nchannels = Table.get("channels", 1);
	print("Images have "+nchannels+" channels");
	run("Close");
	Dialog.create("Channel Configuration Tool");
	choices = newArray("1","2","3","4","5");
	choices = Array.trim(choices, nchannels);
	channels = newArray("nuclei","membrane","granule","other","other other");
	channels = Array.trim(channels, nchannels);
	for (i = 0; i < channels.length; i++) {
		Dialog.addRadioButtonGroup(channels[i], choices, 1, nchannels, "1");
	}
	Dialog.addNumber("scale [um]", 50);
	Dialog.addChoice("Mode", newArray("Cellpose","Cellpose & Watershed"));
	Dialog.show();
	str = "{\"Analysis\":\"SG\", \"channels\":[";
	for (i = 0; i < channels.length; i++) {
		j = parseInt(Dialog.getRadioButton());
		str += "{\"index\":" + j-1 + ", \"name\":\"" + channels[i] + "\"}";
		if (i!= channels.length-1) {
			str+=",";
		}
	}
	scale = Dialog.getNumber();
	if (matches(Dialog.getChoice(), "Cellpose & Watershed")) {
		mode = 1;
	} else {
		mode = 0;
	}
	str += "],\"NA\":0.95,\"medium_refractive_index\":1.4, \"scale_um\":"+scale+", \"mode\":"+mode+"}";
	print("Saving configuration file in folder");
	print(folder+File.separator+"config.json");
	File.saveString(str, folder+File.separator+"config.json");
	print("Configuration file has been created, ready to process the dataset.");
	print(str);
}

function configSpread() {
	print("[ Configuration file for Spread analysis ]");
	print(folder);
	Table.open(folder+"/filelist.csv");
	nchannels = Table.get("channels", 1);
	print("Images have "+nchannels+" channels");
	run("Close");
	Dialog.create("Channel Configuration Tool");

	choices = newArray("1","2","3","4","5");
	choices = Array.trim(choices, nchannels);
	channels = newArray("nuclei","label1","label2","label3","label4");
	channels = Array.trim(channels, nchannels);
	Dialog.addMessage("Indicate the labels in channel order\nwith least one nuclei");
	for (i = 0; i < channels.length; i++) {
		Dialog.addString("Channel "+(i+1), channels[i]);
	}
	Dialog.addNumber("scale [um]", 50);
	Dialog.addChoice("Mode", newArray("Cellpose","Cellpose & Watershed"));
	Dialog.show();
	str = "{\"Analysis\":\"Spread\", \"channels\":[";
	for (i = 0; i < channels.length; i++) {
		chname = Dialog.getString();
		str += "{\"index\":" + i + ", \"name\":\"" + chname + "\"}";
		if (i!= channels.length-1) {
			str+=",";
		}
	}
	scale = Dialog.getNumber();
	if (matches(Dialog.getChoice(), "Cellpose & Watershed")) {
		mode = 1;
	} else {
		mode = 0;
	}
	str += "],\"NA\":0.95,\"medium_refractive_index\":1.4, \"scale_um\":"+scale+", \"mode\":"+mode+"}";
	print("Saving configuration file in folder");
	print(folder+File.separator+"config.json");
	File.saveString(str, folder+File.separator+"config.json");
	print("Configuration file has been created, ready to process the dataset.");
	print(str);
}

function process() {
	print("[ Processing a list of files ]");
	print("Open filelist.csv and create a job for each line.");
	if (!File.exists(folder+File.separator+"results")) {
		print("Creating results directory");
		File.makeDirectory(folder+File.separator+"results");
	}
	jobname = "sga-process.sh";
	str  = "#!/bin/bash\n";
	str += "#SBATCH --job-name=sga-process\n";
	str += "#SBATCH --time=05:00:00\n";
	if (use_gpu_queue) {
		str += "#SBATCH --partition=gpu\n";
		str += "#SBATCH --gres=gpu:1\n";
		str += "#SBATCH -c 12\n";
	} else {
		str += "#SBATCH --partition=cpu\n";
		str += "#SBATCH -c 32\n";
	}
	str += "pwd\n";
	str += "I=$(printf %06d $SLURM_ARRAY_TASK_ID)\n";
	str += "echo $I\n";
	str += cmd +"process --data-path=\""+remote_path+"\" ";
	str += "--file-list \""+remote_path+"/filelist.csv\" --index $I ";
	str += "--output-by-cells \""+remote_path+"\"/results/cells$I.csv ";
	str += "--output-vignette \""+remote_path+"\"/results/vignettes$I.png";

	File.saveString(str,local_jobs_dir+File.separator+jobname);
	Table.open(folder+File.separator+"filelist.csv");
	n = Table.size;
	print("Sending command to "+ hostname);
	ret = exec("ssh", username+"@"+hostname, "sbatch", "--chdir", remote_jobs_dir,  "--array=0-"+(n-1), jobname);
	print(ret);
	jobid = parseInt(replace(ret,"Submitted batch job ",""));
	print("Job is running in the background, use 'List Jobs' to check completion.");
	print(local_jobs_dir + File.separator + "slurm-" + jobid + "_1.out");
	print("Next: list running jobs and once finished make a figure.");
}

function figure() {
	print("[ Preparing a figure ]");
	jobname = "sga-fig.sh";
	str  = "#!/bin/tcsh\n#SBATCH --job-name=sg-fig\n#SBATCH --time=01:00:00\n"+cmd+"figure --data-path=\""+remote_path+"\" --file-list \""+remote_path+"/filelist.csv\"";
	File.saveString(str,local_jobs_dir+File.separator+jobname);
	ret = exec("ssh", username+"@"+hostname, "sbatch", "--chdir", remote_jobs_dir, jobname);
	print(ret);
	jobid = parseInt(replace(ret,"Submitted batch job ",""));
	print("Job is running in the background, use 'List Jobs' to check completion.");
	print(local_jobs_dir + File.separator + "slurm-" + jobid + ".out");
	print("Once the job is completed, opne the files:");
	print(folder+"/results/cells.csv");
	print(folder+"/results/cells.pdf");
}

function listJobs() {
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
		setResult("LOG",i-1,local_jobs_dir + File.separator + "slurm-"+elem[0]+".out");
	}
}

function cancelJobs() {
	print("[ Cancel jobs ]");

	if (isOpen("Results")) { selectWindow("Results"); run("Close"); }
	ret = exec("ssh", username+"@"+hostname, "squeue -u "+username);
	lines = split(ret,"\n");
	if (lines.length == 1) {
		print("No jobs are running");
	}
	str = "";
	for (i = 1; i < lines.length; i++) {
		elem = split(lines[i]," ");
		str += elem[0] + " ";
	}
	ok = getBoolean("Cancel all your jobs on the cluster?\n"+str);
	if (ok) {
		print("Cancelling jobs " + str);
		ret = exec("ssh", username+"@"+hostname, "scancel " + str);
		print(ret);
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
