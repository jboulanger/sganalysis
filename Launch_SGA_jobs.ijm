#@String (label="Username", value="username", description="Username on the host") username
#@String (label="Host", value="hex", description="hostname to which command are send") hostname
#@File(label="Folder", value="", style="directory",description="Path to the data folder from this computer") folder
#@String(label="Action",choices={"Install","Scan","Process","Figure"}) action
#@File (label="Local share",description="Local mounting point of the network share", style="directory") local_share
#@String (label="Remote share",description="Remote mounting point of the network share", value="/cephfs/") remote_share

/*
 * Launch a slurm job with the file filename as input using a template script
 * 
 * Jerome Boulanger 2021 
 */


delay = 500; // artificial delays that prevent denial of service 
remote_path = replace(convert_slash(folder), convert_slash(local_share), remote_share);
remote_jobs_dir = remote_share + "/jobs";
local_jobs_dir = local_share + File.separator + "jobs";
script_url = "https://raw.githubusercontent.com/jboulanger/sganalysis/master/sganalysiswf.py";
sript_name = "sganalysiswf.py";

// create a job folder if needed
if (File.exists(local_jobs_dir) != 1) {
	print("Create a jobs folder in " + local_share);
	File.makeDirectory(local_jobs_dir);
} else {
	print("Jobs folder already present in " + local_share);
}

if (matches(action, "Scan")) {
	scan();
} else if (matches(action, "Process")) {
	process();
}else if (matches(action, "Figure")) {
	figure();	
} else if (matches(action, "Install"))  {
	install();
}

function install() {
	str = File.openUrlAsString(script_url);
	dst = local_jobs_dir + File.separator + "sganalysiswf.py";
	File.saveString(str,local_jobs_dir+File.separator+script_name);	
}

function scan() {
	print("scan");
	jobname = "sga-scan.sh";
	str  = "#!/bin/tcsh\n#SBATCH --job-name=sg-scan\n#SBATCH --time=01:00:00\nconda activate sganalysis\npython sganalysiswf.py scan --data-path=\""+remote_path+"\" --file-list \""+remote_path+"/filelist.csv\"";
	File.saveString(str,local_jobs_dir+File.separator+jobname);		
	ret = exec("ssh", username+"@"+hostname, "sbatch", "--chdir", remote_jobs_dir, jobname);
	print(ret);
}

function process() {
	print("process");
	if (!File.exists(folder+File.separator+"results")) {
		print("Creating result directory");
		File.makeDirectory(folder+File.separator+"results");
	}	
	jobname = "sga-process.sh";
	str  = "#!/bin/tcsh\n#SBATCH --job-name=sga-process\n#SBATCH --time=05:00:00\n#SBATCH --partition=gpu\n#SBATCH --gres=gpu:1\nconda activate sganalysis\npython sganalysiswf.py process --data-path=\""+remote_path+"\" --file-list \""+remote_path+"/filelist.csv\" --index $SLURM_ARRAY_TASK_ID --output-by-cells \""+remote_path+"\"/results/cells-$SLURM_ARRAY_TASK_ID.csv --output-vignette \""+remote_path+"\"/results/vignettes-$SLURM_ARRAY_TASK_ID.png";
	File.saveString(str,local_jobs_dir+File.separator+jobname);
	Table.open(folder+File.separator+"filelist.csv");
	n = Table.size;
	ret = exec("ssh", username+"@"+hostname, "sbatch", "--chdir", remote_jobs_dir,  "--array=1-"+n, jobname);
	print(ret);
}

function figure() {
	print("figure");
	jobname = "sga-scan.sh";
	str  = "#!/bin/tcsh\n#SBATCH --job-name=sg-fig\n#SBATCH --time=01:00:00\nconda activate sganalysis\npython sganalysiswf.py figure --data-path=\""+remote_path+"\" --file-list \""+remote_path+"/filelist.csv\"";
	File.saveString(str,local_jobs_dir+File.separator+jobname);		
	ret = exec("ssh", username+"@"+hostname, "sbatch", "--chdir", remote_jobs_dir, jobname);
	print(ret);	
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