import os
import sys
import logging
import optparse
import xml.etree.ElementTree as ET


logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)

ROOT_PATH = '/vireo00/nikki/AVS_data'
COLLECTION = 'iacc.3'
EDITION = 'tv18'


def parse_result(res):
    resp = {}
    lines = res.split('\n')
    for line in lines:
        elems = line.split()
        if 'infAP' in elems:
            print(line)
        if 'infAP' == elems[0] and 'all' in line:
            return float(elems[-1])


def xml_to_treceval(opt, input_file):
    overwrite = opt.overwrite
    # print(input_file)
    res_file = os.path.splitext(input_file)[0] + '.treceval'

    if os.path.exists(res_file):
        if overwrite:
            logger.info('%s exists. Overwrite' % res_file)
        else:
            logger.info('%s exists. Use "--overwrite 1" if you want to overwrite' % res_file)
            return res_file

    tree = ET.parse(input_file)
    root = tree.getroot()

    MAX_SCORE = 9999
    TEAM = 'VIREO'

    newlines = []

    for topicResult in root.iter('videoAdhocSearchTopicResult'):
        topicResult.attrib['tNum'] = topicResult.attrib['tNum'].replace('manual_','')

        if len(topicResult.attrib['tNum'])<4:
            qry_id = '1' + topicResult.attrib['tNum']
        else:
            qry_id = topicResult.attrib['tNum']

        itemlist = topicResult.getchildren()
        # showline = []
        for rank, item in enumerate(itemlist):
            assert(rank+1 == int(item.attrib['seqNum']))
            shot_id = item.attrib['shotId']
            score = MAX_SCORE - rank
            newlines.append('%s 0 %s %d %d %s' % (qry_id, shot_id, rank+1, score, TEAM))
            # showline.append('%s %d'%(shot_id,score))

        # showoutfile = os.path.splitext(input_file)[0] + '.'+qry_id
        # print(showoutfile)
        # with open(showoutfile,'w') as fout:
        #     fout.write('\n'.join(showline) + '\n')
    fw = open(res_file, 'w')
    fw.write('\n'.join(newlines)+'\n')
    fw.close()

    return res_file


def process(opt, input_xml_file):

    treceval_file = xml_to_treceval(opt, input_xml_file)
    res_file = os.path.join(os.path.dirname(input_xml_file), 'perf.txt')
    gt_file = os.path.join(opt.rootpath, opt.collection, 'TextData', 'avs.qrels.%s' % opt.edition)
    print("gt_file:%s"%gt_file)
    cmd = 'perl sample_eval.pl -q %s %s %d' % (gt_file, treceval_file,opt.topk)
    print(cmd)
    res = os.popen(cmd).read()

    with open(res_file, 'w') as fw:
        fw.write(res)

    resp = parse_result(res)

    print ('%s infAP: %.3f' % (opt.edition, resp))



def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] input_xml_file""")
    parser.add_option('--rootpath', type=str, default=ROOT_PATH, help='path to datasets. (default: %s)'%ROOT_PATH)
    parser.add_option('--collection', type=str, default=COLLECTION, help='test collection')
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    parser.add_option("--edition", default=EDITION, type="string", help="trecvid edition (default: %s)" % EDITION)
    parser.add_option("--topk", default=1000, type="int", help="xinf@ (default: %d)" % 1000)
            
    (options, args) = parser.parse_args(argv)
    if len(args) < 1:
        parser.print_help()
        return 1

    return process(options, args[0])


if __name__ == "__main__":
    sys.exit(main())
