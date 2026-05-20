"""
DaoTi V53 Benchmark Evaluation Script
======================================
Reproducible evaluation of the DaoTi V53 foundation model on structured tasks.

Usage:
    python eval_benchmark.py              # Random token ids (default)
    python eval_benchmark.py --real-text  # Real Chinese text via char-level tokenizer

Outputs:
    - 64-hexagram palace classification accuracy
    - 64-hexagram retrieval top-1 / top-5 accuracy
    - Coherence distribution statistics
    - Per-palace breakdown
"""

import torch
import torch.nn.functional as F
import json
import sys
import os
import argparse
from inference import (
    load_daoti, predict, compute_coherence, verify_sha256,
    generate_response, GUA_64, BA_GONG, GUA_TRIGRAM, GUA_WUXING,
    find_palace, sparse_expand_input, STATE_DIM, MAX_SEQ,
)

PALACE_NAMES  = ["乾宫","坤宫","震宫","巽宫","坎宫","离宫","艮宫","兑宫"]
LIUQIN_NAMES  = ["父母","兄弟","子孙","妻财","官鬼","空亡"]
LIUSHEN_NAMES = ["青龙","朱雀","勾陈","螣蛇","白虎","玄武"]

GUA_REAL_TEXTS = [
    "乾为天，刚健中正，自强不息。天行健，君子以自强不息。元亨利贞，四德具备。乾道变化，各正性命。",
    "坤为地，厚德载物，柔顺包容。地势坤，君子以厚德载物。元亨，利牝马之贞。坤道其顺，承天而时行。",
    "屯卦，水雷屯，万物初生，艰难险阻。云雷屯，君子以经纶。刚柔始交而难生，动乎险中。",
    "蒙卦，山水蒙，启蒙教化，蒙以养正。山下出泉，蒙，君子以果行育德。匪我求童蒙，童蒙求我。",
    "需卦，水天需，守正待时，诚信则通。云上于天，需，君子以饮食宴乐。有孚光亨贞吉。",
    "讼卦，天水讼，争讼不和，终凶。天与水违行，讼，君子以作事谋始。有孚窒惕中吉终凶。",
    "师卦，地水师，兴师动众，纪律严明。地中有水，师，君子以容民畜众。贞丈人吉无咎。",
    "比卦，水地比，亲比相辅，择善而从。地上有水，比，先王以建万国亲诸侯。吉原筮元永贞无咎。",
    "小畜卦，风天小畜，蓄积微弱，以柔济刚。风行天上，小畜，君子以懿文德。亨密云不雨。",
    "履卦，天泽履，循礼而行，慎始敬终。上天下泽，履，君子以辨上下定民志。履虎尾不咥人亨。",
    "泰卦，地天泰，天地交泰，万物通达。天地交，泰，后以财成天地之道。小往大来吉亨。",
    "否卦，天地否，天地不交，闭塞不通。天地不交，否，君子以俭德辟难。大往小来否之匪人。",
    "同人卦，天火同人，和同于人，志同道合。天与火，同人，君子以类族辨物。同人于野亨。",
    "大有卦，火天大有，日丽中天，大有收获。火在天上，大有，君子以遏恶扬善顺天休命。元亨。",
    "谦卦，地山谦，谦虚受益，卑以自牧。地中有山，谦，君子以裒多益寡称物平施。亨君子有终。",
    "豫卦，雷地豫，愉悦安乐，居安思危。雷出地奋，豫，先王以作乐崇德殷荐之上帝。利建侯行师。",
    "随卦，泽雷随，随时变通，择善而从。泽中有雷，随，君子以向晦入宴息。元亨利贞无咎。",
    "蛊卦，山风蛊，整治败坏，拨乱反正。山下有风，蛊，君子以振民育德。元亨利涉大川。",
    "临卦，地泽临，居高临下，亲临指导。泽上有地，临，君子以教思无穷容保民无疆。元亨利贞。",
    "观卦，风地观，观察审视，以道示人。风行地上，观，先王以省方观民设教。盥而不荐有孚颙若。",
    "噬嗑卦，火雷噬嗑，明罚敕法，决断是非。雷电噬嗑，先王以明罚敕法。亨利用狱。",
    "贲卦，山火贲，文饰美化，文质彬彬。山下有火，贲，君子以明庶政无敢折狱。亨小利有攸往。",
    "剥卦，山地剥，剥落侵蚀，顺时而止。山附于地，剥，上以厚下安宅。不利有攸往。",
    "复卦，地雷复，一阳来复，返本归元。雷在地中，复，先王以至日闭关商旅不行。亨出入无疾朋来无咎。",
    "无妄卦，天雷无妄，至诚不妄，顺应天道。天下雷行物与无妄，先王以茂对时育万物。元亨利贞。",
    "大畜卦，山天大畜，蓄养贤能，积厚流光。天在山中，大畜，君子以多识前言往行以畜其德。利贞。",
    "颐卦，山雷颐，养正则吉，慎言节食。山下有雷，颐，君子以慎言语节饮食。贞吉观颐自求口实。",
    "大过卦，泽风大过，非常之时，独立不惧。泽灭木大过，君子以独立不惧遁世无闷。栋桡利有攸往亨。",
    "坎卦，坎为水，重险重重，诚信可济。水洊至习坎，君子以常德行习教事。有孚维心亨行有尚。",
    "离卦，离为火，光明附丽，柔顺中正。明两作离，大人以继明照于四方。利贞亨畜牝牛吉。",
    "咸卦，泽山咸，感应相通，以虚受人。山上有泽咸，君子以虚受人。亨利贞取女吉。",
    "恒卦，雷风恒，持之以恒，久于其道。雷风恒，君子以立不易方。亨无咎利贞利有攸往。",
    "遁卦，天山遁，退避隐遁，明哲保身。天下有山遁，君子以远小人不恶而严。亨小利贞。",
    "大壮卦，雷天大壮，阳刚壮盛，非礼弗履。雷在天上大壮，君子以非礼弗履。利贞。",
    "晋卦，火地晋，光明上进，顺而丽明。明出地上晋，君子以自昭明德。康侯用锡马蕃庶昼日三接。",
    "明夷卦，地火明夷，光明受伤，韬光养晦。明入地中明夷，君子以莅众用晦而明。利艰贞。",
    "家人卦，风火家人，治家之道，正位居体。风自火出家人，君子以言有物而行有恒。利女贞。",
    "睽卦，火泽睽，乖异背离，求同存异。上火下泽睽，君子以同而异。小事吉。",
    "蹇卦，水山蹇，行路艰难，见险而止。山上有水蹇，君子以反身修德。利西南不利东北利见大人贞吉。",
    "解卦，雷水解，解除困难，百事舒缓。雷雨作解，君子以赦过宥罪。利西南无所往其来复吉。",
    "损卦，山泽损，减损自我，损下益上。山下有泽损，君子以惩忿窒欲。有孚元吉无咎可贞利有攸往。",
    "益卦，风雷益，增益上进，损上益下。风雷益，君子以见善则迁有过则改。利有攸往利涉大川。",
    "夬卦，泽天夬，决断刚毅，防微杜渐。泽上于天夬，君子以施禄及下居德则忌。扬于王庭孚号有厉。",
    "姤卦，天风姤，不期而遇，防一阴生。天下有风姤，后以施命诰四方。女壮勿用取女。",
    "萃卦，泽地萃，聚集汇合，以正聚众。泽上于地萃，君子以除戎器戒不虞。亨王假有庙利见大人。",
    "升卦，地风升，上升进取，积小成大。地中生木升，君子以顺德积小以高大。元亨用见大人勿恤。",
    "困卦，泽水困，困境穷厄，守正待时。泽无水困，君子以致命遂志。亨贞大人吉无咎有言不信。",
    "井卦，水风井，养民不穷，修德不已。木上有水井，君子以劳民劝相。改邑不改井无丧无得。",
    "革卦，泽火革，变革更新，顺天应人。泽中有火革，君子以治历明时。已日乃孚元亨利贞悔亡。",
    "鼎卦，火风鼎，革故鼎新，正位凝命。木上有火鼎，君子以正位凝命。元吉亨。",
    "震卦，震为雷，奋发激荡，恐惧修省。洊雷震，君子以恐惧修省。亨震来虩虩笑言哑哑。",
    "艮卦，艮为山，止而不动，知止而止。兼山艮，君子以思不出其位。艮其背不获其身行其庭不见其人无咎。",
    "渐卦，风山渐，循序渐进，以正合礼。山上有木渐，君子以居贤德善俗。女归吉利贞。",
    "归妹卦，雷泽归妹，归嫁之象，待时而动。泽上有雷归妹，君子以永终知敝。征凶无攸利。",
    "丰卦，雷火丰，丰盛盈满，明以动之。雷电皆至丰，君子以折狱致刑。亨王假之勿忧宜日中。",
    "旅卦，火山旅，旅途羁旅，柔顺中正。山上有火旅，君子以明慎用刑而不留狱。小亨旅贞吉。",
    "巽卦，巽为风，随风顺入，柔渗透达。随风巽，君子以申命行事。小亨利有攸往利见大人。",
    "兑卦，兑为泽，喜悦和乐，朋友讲习。丽泽兑，君子以朋友讲习。亨利贞。",
    "涣卦，风水涣，涣散离析，以正聚合。风行水上涣，先王以享于帝立庙。亨王假有庙利涉大川利贞。",
    "节卦，水泽节，节制适度，制度合礼。泽上有水节，君子以制数度议德行。亨苦节不可贞。",
    "中孚卦，风泽中孚，诚信为本，信及豚鱼。泽上有风中孚，君子以议狱缓死。豚鱼吉利涉大川利贞。",
    "小过卦，雷山小过，小有过越，行止得宜。山上有雷小过，君子以行过乎恭丧过乎哀用过乎俭。亨利贞可小事不可大事。",
    "既济卦，水火既济，事已成就，守成防衰。水在火上既济，君子以思患而豫防之。亨小利贞初吉终乱。",
    "未济卦，火水未济，事未完成，审慎而行。火在水上未济，君子以慎辨物居方。亨小狐汔济濡其尾无攸利。",
]

def simple_chinese_tokenize(text, vocab_size=8145, max_seq=256):
    tokens = []
    for ch in text:
        idx = ord(ch) % vocab_size
        if idx == 0:
            idx = 1
        tokens.append(idx)
    if len(tokens) < max_seq:
        tokens = tokens + [0] * (max_seq - len(tokens))
    else:
        tokens = tokens[:max_seq]
    return torch.tensor([tokens], dtype=torch.long)

def eval_palace_classification(model, device='cpu', text_mode='random'):
    correct = 0
    total = 64
    per_palace = {p: {"correct": 0, "total": 0} for p in PALACE_NAMES}

    for gi in range(64):
        if text_mode == 'real':
            text_ids = simple_chinese_tokenize(GUA_REAL_TEXTS[gi]).to(device)
        else:
            text_ids = torch.randint(1, 100, (1, 256), dtype=torch.long, device=device)
        gua_name = GUA_64[gi]
        ground_truth = find_palace(gua_name)
        r = predict(model, text_ids, gua_idx=gi, method='traditional', device=device)
        pred_palace = PALACE_NAMES[r['palace'].argmax().item()]
        per_palace[ground_truth]["total"] += 1
        if pred_palace == ground_truth:
            correct += 1
            per_palace[ground_truth]["correct"] += 1

    accuracy = correct / total
    return {
        "task": "64-Hexagram Palace Classification",
        "accuracy": f"{100*accuracy:.1f}%",
        "correct": correct,
        "total": total,
        "per_palace": {p: f"{v['correct']}/{v['total']}" for p, v in per_palace.items()},
    }

def eval_retrieval(model, device='cpu', text_mode='random'):
    with torch.no_grad():
        proto = model.gua_prototype.weight
        proto_n = F.normalize(proto, p=2, dim=-1)

    top1_correct = 0
    top5_correct = 0
    similarities = []

    for gi in range(64):
        if text_mode == 'real':
            text_ids = simple_chinese_tokenize(GUA_REAL_TEXTS[gi]).to(device)
        else:
            text_ids = torch.randint(1, 100, (1, 256), dtype=torch.long, device=device)
        with torch.no_grad():
            text_feat = model.encode_text(text_ids.to(device))
            feat_n = F.normalize(text_feat, p=2, dim=-1)
            sim = torch.mm(feat_n, proto_n.t()).squeeze()
            similarities.append(sim.max().item())
            top1_idx = sim.argmax().item()
            top5_idx = sim.topk(5).indices.tolist()
        if top1_idx == gi:
            top1_correct += 1
        if gi in top5_idx:
            top5_correct += 1

    return {
        "task": "64-Hexagram Prototype Retrieval",
        "top1_accuracy": f"{100*top1_correct/64:.1f}%",
        "top5_accuracy": f"{100*top5_correct/64:.1f}%",
        "top1_correct": top1_correct,
        "top5_correct": top5_correct,
        "total": 64,
        "avg_similarity": f"{sum(similarities)/len(similarities):.4f}",
        "min_similarity": f"{min(similarities):.4f}",
        "max_similarity": f"{max(similarities):.4f}",
    }

def eval_coherence_distribution(model, device='cpu', text_mode='random'):
    coherences = []
    for gi in range(64):
        if text_mode == 'real':
            text_ids = simple_chinese_tokenize(GUA_REAL_TEXTS[gi]).to(device)
        else:
            text_ids = torch.randint(1, 100, (1, 256), dtype=torch.long, device=device)
        c = compute_coherence(model, text_ids, gi, device)
        coherences.append(c)

    coherences.sort()
    n = len(coherences)
    return {
        "task": "Coherence Distribution (Self-Calibrating Quality Signal)",
        "mean": f"{sum(coherences)/n:.4f}",
        "median": f"{coherences[n//2]:.4f}",
        "min": f"{min(coherences):.4f}",
        "max": f"{max(coherences):.4f}",
        "p25": f"{coherences[n//4]:.4f}",
        "p75": f"{coherences[3*n//4]:.4f}",
        "below_threshold_0.3": sum(1 for c in coherences if c < 0.3),
        "above_threshold_0.7": sum(1 for c in coherences if c > 0.7),
    }

def eval_random_baseline(device='cpu'):
    import random
    correct = 0
    for gi in range(64):
        ground_truth = find_palace(GUA_64[gi])
        random_palace = random.choice(PALACE_NAMES)
        if random_palace == ground_truth:
            correct += 1
    return {
        "task": "Random Baseline (8-class uniform)",
        "expected_accuracy": "12.5%",
        "sampled_accuracy": f"{100*correct/64:.1f}%",
    }

def main():
    parser = argparse.ArgumentParser(description="DaoTi V53 Benchmark Evaluation")
    parser.add_argument('--real-text', action='store_true',
                        help='Use real Chinese text (char-level tokenizer) instead of random token ids')
    args = parser.parse_args()

    text_mode = 'real' if args.real_text else 'random'
    mode_label = "Real Chinese Text" if text_mode == 'real' else "Random Token IDs"

    print("=" * 60)
    print(f"  DaoTi V53 Benchmark Evaluation")
    print(f"  Text Mode: {mode_label}")
    print("=" * 60)

    if not verify_sha256("yijing_v53_daoti.pt"):
        print("[FAIL] Weight verification failed")
        sys.exit(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_daoti("yijing_v53_daoti.pt", device=device)

    results = {}

    print("\n[1/4] Random Baseline...")
    results["random_baseline"] = eval_random_baseline(device)
    for k, v in results["random_baseline"].items():
        if k != "task": print(f"  {k}: {v}")

    print("\n[2/4] Palace Classification...")
    results["palace_classification"] = eval_palace_classification(model, device, text_mode=text_mode)
    for k, v in results["palace_classification"].items():
        if k != "task": print(f"  {k}: {v}")

    print("\n[3/4] Prototype Retrieval...")
    results["retrieval"] = eval_retrieval(model, device, text_mode=text_mode)
    for k, v in results["retrieval"].items():
        if k != "task": print(f"  {k}: {v}")

    print("\n[4/4] Coherence Distribution...")
    results["coherence"] = eval_coherence_distribution(model, device, text_mode=text_mode)
    for k, v in results["coherence"].items():
        if k != "task": print(f"  {k}: {v}")

    output_path = "benchmark_results_real_text.json" if text_mode == 'real' else "benchmark_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
