import sys
import glob
import os.path

'''
Scirpt to give utterance-level WER :
Input:
list of decode dirs

Output:
WER computed from 
number of insertions deletions and subtractions

'''

## functions

def get_score_prf(prf_file, cond_list, scores_dict):

    score_flag=False
    
    with open(prf_file) as f:
        for line in f:
            # split line and check first field if 'File:'
            l = line.split()

            if len(l) != 0:
                if l[0] == "File:":
                    if l[1].split('_')[0] in cond_list:
                        score_flag=True
            
                if score_flag:                
                    if l[0] == "Scores:":
                        score_flag=False
                        scores_dict['C'] = scores_dict['C']+int(l[5])
                        scores_dict['S'] = scores_dict['S']+int(l[6])
                        scores_dict['D'] = scores_dict['D']+int(l[7])
                        scores_dict['I'] = scores_dict['I']+int(l[8])

                    
def find_lmwt_span(decode_dir):
    to_glob=decode_dir+"/score*"
    score_list=glob.glob(to_glob)
    
    lmwt_list=[]
    for s in score_list:
        lmwt_list.append(int(os.path.basename(s).split('_')[1]))
        
    lmwt_list=sorted(lmwt_list)
    
    return lmwt_list

def get_wer_frm_CISD(scores_dict):
    return "%.1f" % (float((scores_dict['S'] + scores_dict['D'] + scores_dict['I']))/float((scores_dict['S'] + scores_dict['D'] + scores_dict['C'] ))*100)

def get_score(expt_dir, cond_list, x):
    
    #  get decode_dirs from expt dir
    to_glob=expt_dir[0]+"/decode*"+x
    decode_dirs = glob.glob(to_glob)
    
    if len(decode_dirs) == 0:
        print "can't find decode dir in "
        print to_glob
        sys.exit(1)



    lmwt_span=find_lmwt_span(decode_dirs[0])
    min_wer=100;
    for lmwt in lmwt_span:
        scores_dict = {'C': 0, 'S': 0, 'D': 0, 'I': 0}
        for d in decode_dirs:
            to_glob=d+"/score_"+str(lmwt)+"/*.prf"
            prf_file=glob.glob(to_glob)
            get_score_prf(prf_file[0], cond_list, scores_dict)

        this_wer=get_wer_frm_CISD(scores_dict)
        this_wer=float(this_wer)

        if this_wer < min_wer:
            min_wer=this_wer
            
        # print lmwt, min_wer, this_wer
            
    return min_wer



# main
noise_grp_cond_list=[["clean"], \
                     ["babble-00","babble-05","babble-10","babble-15","babble-20"], \
                     ["benz-00","benz-05","benz-10","benz-15","benz-20"], \
                     ["buccaneer1-00","buccaneer1-05","buccaneer1-10","buccaneer1-15","buccaneer1-20" ], \
                     ["buccaneer2-00","buccaneer2-05","buccaneer2-10","buccaneer2-15","buccaneer2-20"], \
                     ["car-00","car-05","car-10","car-15","car-20"], \
                     ["destroyerops-00","destroyerops-05","destroyerops-10","destroyerops-15","destroyerops-20"], \
                     ["exhall-00","exhall-05","exhall-10","exhall-15","exhall-20"], \
                     ["f16-00","f16-05","f16-10","f16-15","f16-20"], \
                     ["factory1-00","factory1-05","factory1-10","factory1-15","factory1-20"], \
                     ["factory2-00","factory2-05","factory2-10","factory2-15","factory2-20"], \
                     ["leopard-00","leopard-05","leopard-10","leopard-15","leopard-20"], \
                     ["m109-00","m109-05","m109-10","m109-15","m109-20"], \
                     ["restuarant-00","restuarant-05","restuarant-10","restuarant-15","restuarant-20"], \
                     ["street-00","street-05","street-10","street-15","street-20"], \
                     ["subway-00","subway-05","subway-10","subway-15","subway-20"]]

expt_list=["mix3.5hr_new"]


fo=open("score.log", "w")
for cond_list in noise_grp_cond_list:
    print "Noise condition: ", cond_list[0]
    print "--  --------------------------------"
    
    fo.write("Noise condition: "+cond_list[0]+"\n"+"--  --------------------------------\n")
    
    for expt in expt_list:
        decode_dirs=['dnn_exp/exp_' +expt+ '/dnn5tri_splice10_pretrain-dbn_dnn/']
        wer = get_score(decode_dirs, cond_list, "dev")
        
        print wer
        fo.write(expt+" "+str(wer)+"\n")
    

    os.system("pause")

fo.close()



