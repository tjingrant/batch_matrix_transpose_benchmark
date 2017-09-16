import itertools
import subprocess

batch_num = [2**i for i in range(5, 13)]
matrix_height = range(96, 2048, 16)
matrix_width  = range(2, 16)

def extract_avg_time(keyword):
	logs = list(filter(lambda x: x.find(keyword)>0, output.split("\n")))
	avg_time = (list(filter(len, logs[0].split(" "))))[3].strip()

	if avg_time.endswith("us"):
		avg_time = float(avg_time.replace("us", ""))
	elif avg_time.endswith("ms"):
		avg_time = float(avg_time.replace("ms", "")) * 1000
	elif avg_time.endswith("s"):
		avg_time = float(avg_time.replace("s", "")) * 1000000
	return avg_time

for bn in batch_num:
	for mh in matrix_height:
		for mw in matrix_width:
			cmd = "nvprof ./transpose " + str(bn) + " " + str(mh) + " " + str(mw)
			try:
				output = subprocess.check_output(
					cmd, stderr=subprocess.STDOUT, shell=True, timeout=100,
					universal_newlines=True)
			except subprocess.CalledProcessError as exc:
				print("Status : FAIL", exc.returncode, exc.output)
			else:
				new_time = extract_avg_time("void MySwapDimension");
				baseline_time = extract_avg_time("void SwapDimension");
				print(new_time/baseline_time, end='\t', flush=True)
		print("")
