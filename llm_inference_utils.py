from dataclasses import dataclass
from typing import List

import torch
from tqdm import tqdm
import transformers

from io_utils import cont_write_textfile


@dataclass
class Examples:
    source: str
    target: str


@dataclass
class LLMTask:
    description: str
    demonstrations: List[Examples]
    template: str

    def build_prompt(self, num_examples):
        prompt = [self.description] if len(self.description) else []
        for example in self.demonstrations[:num_examples]:
            prompt += [self.template.format(example.source) + f' {example.target}']
        return "\n".join(prompt)


def sample_llm(source_data: list[str], tokenizer: transformers.models, model: transformers.models, device: str,
               batch_size: int, decode_algo: str, num_sequences: int, gen_filename: str, temperature: float = 0.6,
               topp: float = 0.9, max_tokens: int = 200, lang_id: int = None):
    for i in tqdm(range(0, len(source_data), batch_size)):
        source_batch = source_data[i:i + batch_size]
        inputs = tokenizer(source_batch, padding=True, return_tensors="pt", truncation=True).to(device)
        if decode_algo == 'greedy':
            with torch.no_grad():
                if lang_id:
                    gen_tokens = model.generate(**inputs, do_sample=False, max_new_tokens=max_tokens,
                                                forced_bos_token_id=lang_id, num_beams=1)
                else:
                    gen_tokens = model.generate(**inputs, do_sample=False, max_new_tokens=max_tokens, )
        elif decode_algo == 'beam':
            with torch.no_grad():
                if lang_id:
                    gen_tokens = model.generate(**inputs, do_sample=False, num_beams=5, max_new_tokens=max_tokens,
                                                forced_bos_token_id=lang_id)
                else:
                    gen_tokens = model.generate(**inputs, do_sample=False, num_beams=5, max_new_tokens=max_tokens)
        else:
            with torch.no_grad():
                if lang_id:
                    gen_tokens = model.generate(**inputs, do_sample=True, epsilon_cutoff=0.02, temperature=temperature,
                                                max_new_tokens=max_tokens, num_return_sequences=num_sequences,
                                                forced_bos_token_id=lang_id, num_beams=1)
                else:
                    gen_tokens = model.generate(**inputs, do_sample=True, max_new_tokens=max_tokens,
                                                num_return_sequences=num_sequences, temperature=temperature, top_p=topp)
        gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        answers = [text.split(source_batch[j // num_sequences])[-1].split("\n")[0].strip() for j, text in
                   enumerate(gen_text)]

        cont_write_textfile(answers, gen_filename)

    return


TRANSLATION_DESCR = ""
ENDE_TRANSLATION_EXAMPLES = [
    Examples(
        source="The disease has killed nearly 50 people and infected more than 1,400 in Tunisia.",
        target="Die Krankheit hat beinahe 50 Menschen getötet und mehr als 1400 Menschen in Tunesien infiziert."
    ),
    Examples(
        source="Landray said this failure was particularly exasperating when it came to the use of convalescent plasma, which many doctors believe could have a key role to play in treating seriously ill Covid-19 patients.",
        target="Landray beklagt dieses Versagen besonders, wenn es um die Verwendung von rekonvaleszentem Plasma geht, dem laut Meinung vieler Mediziner eine wichtige Rolle bei der Behandlung ernsthaft kranker Covid-19-Patienten zukomme."
    ),
    Examples(
        source="Daily cases that numbered in the hundreds dropped to low double digits.",
        target="Die täglichen Fälle, die sich auf hunderte beliefen, sanken auf zweistellige Zahlen ab."
    ),
    Examples(
        source="However, a recent poll put West at two percent nationwide, neck and neck with the Libertarian Party's Jo Jorgensen and a point ahead of the Green Party's Howie Hawkins.",
        target="Jedoch lag West bei einer kürzlich erfolgten Befragung landesweit bei zwei Prozent, Kopf an Kopf mit Jo Jorgensen von der Libertarian Party und einen Punkt vor Howie Hawkins von der Green Party."
    ),
    Examples(
        source="Scotland's festival scene and sporting events such as the Highland games have been among those affected by restrictions brought in to prevent the spread of Covid-19.",
        target="Schottlands Festival- und Sporteventszene, wie die Highland Games, waren unter jenen die von den Einschränkungen, welche eingeführt worden sind um die Ausbreitung von Covid-19 zu verhindern, betroffen waren."
    ),
    Examples(
        source="Late Saturday, the U.S. State Department issued a statement saying its peace envoy Zalmay Khalilzad was again shuttling through the region seeking to jump start those negotiations, which have been repeatedly postponed as both sides squabble over a prisoner release program.",
        target="Am späten Samstag, veröffentliche das US- Außenministerium ein Statement, in dem es hieß, dass ihr Friedensbeauftragter Zalmay Khalilzad erneut durch die Region fuhr, um diese Verhandlungen zu beschleunigen, welche wiederholt aufgeschoben wurden, da beide Seiten über ein Gefangenen-Freilassungsprogramm streiten."
    ),
    Examples(
        source="The Union Chain Bridge crosses the River Tweed from Fishwick in the Scottish Borders to Horncliffe in Northumberland.",
        target="Die Union Chain Brücke überquert den Fluss Tweed von Fishwick an der schottischen Grenze nach Horncliffe in Northumberland."
    ),
    Examples(
        source="The pair had their masks off to eat, when a \"random old lady\" approached, calling them \"idiots\" for not masking up, even though \"you can't wear a mask and eat at the same time.\"",
        target="Als die beiden die Masken abnahmen, um zu essen, näherte sich „irgendeine alte Frau“, bezeichnete sie als „Idioten“, weil sie keine Masken trugen, obwohl man „nicht gleichzeitig eine Maske tragen und essen kann“."
    ),
]

ZHEN_TRANSLATION_EXAMPLES = [
    Examples(
        source="乌鲁木齐市卫健委主任张卫26日称，乌鲁木齐全市免费核酸检测工作多数区已基本完成。",
        target="Zhang Wei, director of the Municipal Health Commission of Urumqi, reported that free nucleic acid testing work has basically been completed in most areas of Urumqi."
    ),
    Examples(
        source="美国人口普查将决定美国国会众议院席位的分配。",
        target="The US Census decides the allocation of seats in the US House of Representatives."
    ),
    Examples(
        source="按计划，卡-52M的飞行员能直接操控与其配合的无人机，从而提高飞行员对战场形势的掌握和对远程导弹的制导。",
        target="According to the plan, the Kamov Ka-52M pilot will directly operate the coordinated UAV’s to improve their command of the battlefield and the guidance of long-distance missiles."
    ),
    Examples(
        source="全台各地最大震度为：花莲县4级，宜兰县、台东县、南投县、云林县3级，另有其他县市测得1到2级震度。",
        target="Across Taiwan, the highest earthquake intensity was felt in Hualien County at Level 4, Yilan County, Taitung County, Nantou County, and Yunlin County at 3, and other counties and cities between 1 to 2."
    ),
    Examples(
        source="随着各项稳就业政策落实落细，企业用工需求稳步回升，就业形势逐渐改善，城镇调查失业率逐步下降，6月份，调查失业率为5.7%。",
        target="As employment stabilization policies were implemented, companies demands for labor gradually recovered, and as employment improves, the urban unemployment rate has decreases; in June, the unemployment rate was 5.7%."
    ),
    Examples(
        source="为应对巢湖持续高水位压力和可能的强降雨，安徽省合肥市准备在26日启用傅昆宝家所在的蒋口河联圩分洪蓄水。",
        target="To cope with the continuous high water levels at Chaohu Lake and the possible strong downpour, Hefei City is preparing to use a joint embankment at Jiangkou River, where Fu Kunbao’s home is located, to diverge the flood and save water."
    ),
    Examples(
        source="疫情期间，京演集团旗下院团“停演不停‘艺’，停工不停‘功’”储备佳作。",
        target="During the pandemic, troupes under the group “stopped its shows but never ended its ‘art’, stopped their work but never quit ‘practicing’”, and have prepared dazzling works of art for the future."
    ),
    Examples(
        source="警方表示，目前，警方正在与瑞士运输安全调查委员会合作，共同调查这起坠机事故原因。",
        target="The police said that presently, they are cooperating with the Swiss Transportation Safety Investigation Board to discover the cause of the accident."
    ),
]

ENRU_TRANSLATION_EXAMPLES = [
    Examples(
        source="According to a study prepared by the National Council for Childhood and Motherhood, children are subjected to violence in places that are supposed to be safe, such as home, school or clubs, and exposed to violence from people who are supposed to care for them, such as parents or teachers.",
        target="Согласно исследованию, подготовленному Национальным Советом по Делам Детства и Материнства, дети подвергаются насилию в местах, которые должны быть безопасными, таких как дом, школа или клубы, и становятся жертвами насилия со стороны людей, которые должны о них заботиться, таких как родители или учителя."
    ),
    Examples(
        source="Daily cases that numbered in the hundreds dropped to low double digits.",
        target="Ежедневные случаи сократились с сотен до малых двузначных чисел."
    ),
    Examples(
        source="OAKLAND, Ca. -- Protesters in California set fire to a courthouse, damaged a police station and assaulted officers after a peaceful demonstration intensified late Saturday, Oakland police said.",
        target="ОКЛЕНД, Калифорния -- Как сообщила полиция Окленда, вечером в субботу после того, как мирная демонстрация превратилась в беспорядки, протестующие в Калифорнии подожгли здание суда, нанесли ущерб полицейскому участку и напали на офицеров."
    ),
    Examples(
        source="An unlawful assembly was declared and police ordered protesters to leave the area, authorities said.",
        target="По заявлению властей, собрание было объявлено незаконным, и полицейские приказали протестующим покинуть район."
    ),
    Examples(
        source="Man found with gunshot wounds in Mambourin near Werribee",
        target="Мужчина с огнестрельными ранениями обнаружен в Мамбурине поблизости от Верриби"
    ),
    Examples(
        source="He said: \"It's not something I want to do. A lot of people watching might think that's exactly what he should do and stop talking about things, but I care about the country.\"",
        target="Он сказал: \"Это не просто моя прихоть. Многие, кто нас смотрит, возможно именно так и думают, считают, что так и надо, и мне пора прекратить лезть со своим мнением, но мне небезразлична судьба страны.\""
    ),
    Examples(
        source="In particular, convalescent plasma (blood plasma that is taken from Covid-19 patients and which contains antibodies that could protect others against the disease) has still to be properly tested on a large-scale randomised trial.",
        target="В частности, плазма выздоравливающих пациентов (плазма крови, взятая у больных COVID-19 и содержащая антитела, вероятно, способные защитить от болезни других) все еще ожидает серьезного крупномасштабного исследования методом случайной выборки."
    ),
    Examples(
        source="He paid tribute to his team-mates in receiving the award and said there are still plenty of miles left in him at the top level, despite his advancing age.",
        target="Получив награду, он поблагодарил товарищей по команде и сказал, что у него еще остался порох в пороховницах, чтобы выступать на высшем уровне, несмотря на возраст."
    ),
]

ISEN_TRANSLATION_EXAMPLES = [
    Examples(
        source="En snúum okkur að flækingunum.",
        target="But let's talk about the stray birds."
    ),
    Examples(
        source="Eftir að Icelandair sleit viðræðum við Flugfreyjufélag Íslands og sagði upp öllum flugfreyjum lýsti stjórn VR því yfir að stjórnarmenn sem VR skipaði í stjórn Lífeyrissjóðs verslunarmanna sniðgengju eða greiddu atkvæði gegn þátttöku lífeyrissjóðsins í væntanlegu hlutafjárútboði Icelandair.",
        target="When Icelandair ceased negotiating with the Icelandic Cabin Crew Association and laid off all airline hostesses, the Board of VR declared that VR-appointed members of the Board of the Pension Fund of Commerce would boycott or vote against the pension fund's participation in Icelandair's upcoming stock offering."
    ),
    Examples(
        source="Þú talar um að þið upplifið að þið hafið misst af einhverju, en það er mikilvægt að minna sig á að stórum breytingum eins og skilnaði og nýju sambandi fylgja líka fylgifiskar sem þið eruð ekkert endilega spennt fyrir.",
        target="You say that you feel you have missed out on something, but it is important to remember that big changes, such as divorce and a new relationship, also have side effects that you may not find too attractive."
    ),
    Examples(
        source="„Annars hefur verið tiltölulega rólegt í sumar í flækingunum,“ segir Brynjúlfur en nefnir að í sveitinni sé nú að finna grátrönu og þá hafi hringdúfur sést og séu farnar að verpa hér.",
        target="\"But this summer has actually been relatively quiet when it comes to strays,\" says Brynjúlfur, but notes that a common crane can now be found in the area, and that common wood pigeons have been spotted and are beginning to nest here."
    ),
    Examples(
        source="„Já, það er algjörlega óþolandi, þetta er ein umferðarteppa hérna og ekki bara á föstudögum því þetta er bara orðið á virkum dögum líka.",
        target="\"Yes, it's completely intolerable. This is one big traffic jam here, and not only on Fridays because now it's also on working days."
    ),
    Examples(
        source="„Þess vegna skiptir svo miklu máli að ráða viðeigandi leikara í viðeigandi hlutverk.“",
        target="\"That's why it's imperative that we cast the appropriate actor for the appropriate role.\""
    ),
    Examples(
        source="Undrandi samþykkir vinurinn það og nær í annan starfsmann.",
        target="Surprised, the friend agreed and fetched another crew member."
    ),
    Examples(
        source="Í febrúar 2019 undirritaði ríkisstjórnin friðarsamkomulag við 14 vopnaða hópa sem yfirleitt segjast verja hagsmuni tiltekinna samfélaga eða trúarbragða.",
        target="The government signed a peace deal in February 2019 with 14 armed groups, who typically claim to defend the interests of specific communities or religions."
    ),
]

NLEN_TRANSLATION_EXAMPLES = [
    Examples(
        source="Er zijn diverse restaurants in de buurt van de tuin. 's Middags en 's avonds worden er gratis concerten gegeven in het centraal gelegen prieel.",
        target="There are a number of restaurants surrounding the garden, and in the afternoons and evening there free concerts are often given from the central gazebo."
    ),
    Examples(
        source="Als je reist met een laptop of tablet, is het handig om een kopie van het geheugen of de schijf te bewaren (waar het toegankelijk is zonder internet).",
        target="If traveling with a laptop or tablet, store a copy in its memory or disc (accessible without the internet)."
    ),
    Examples(
        source="Producten kunnen zo nodig worden gekocht, maar de meeste producten hebben weinig tot geen echte impact op de prestaties.",
        target="Products can be purchased as needed, but most will have little or no real impact on performance."
    ),
    Examples(
        source="De stalen naald drijft dankzij de oppervlaktespanning op het water.",
        target="The steel needle floats on top of the water because of surface tension."
    ),
    Examples(
        source="Dankzij het uitvinden van het spaakwiel werden de strijdwagens van Assyrië lichter, sneller en daardoor beter uitgerust voor het ontlopen van soldaten en andere strijdwagens.",
        target="The invention of spoke wheels made Assyrian chariots lighter, faster, and better prepared to outrun soldiers and other chariots."
    ),
    Examples(
        source="Eerder leed hij een nederlaag tegen Raonic in de Brisbane Open.",
        target="He recently lost against Raonic in the Brisbane Open."
    ),
    Examples(
        source="In het meest recente rapport verklaarde OPEC dat de export van ruwe olie op het dieptepunt van de afgelopen twee decennia is beland met 2,8 miljoen vaten per dag.",
        target="In its most recent monthly report, OPEC said exports of crude had fallen to their lowest level for two decades at 2.8 million barrels per day."
    ),
    Examples(
        source="Een tweede beperkingsgebied onder de tanks dat tot 104.500 vaten kan opslaan, was nog niet volledig gevuld.",
        target="Another secondary containment area below the tanks capable of holding 104,500 barrels was not yet filled to capacity."
    ),
]

DEFR_TRANSLATION_EXAMPLES = [
    Examples(
        source="Trump bewirbt sich bei der Wahl im November für eine zweite Amtszeit.",
        target="Trump est candidat à un deuxième mandat aux élections de novembre."
    ),
    Examples(
        source="In der betreffenden Arbeitsgruppe ging es um den heftigen Streit mit der Zulieferergruppe Prevent vor einigen Jahren - und wie Volkswagen darauf reagieren wollte.",
        target="Dans le groupe de travail concerné, il s’agissait du violent litige survenu avec le groupe de fournisseurs Prevent il y a quelques années et de la manière dont Volkswagen avait voulu y réagir."
    ),
    Examples(
        source="Das stellt der Arbeitgeberverband (AGV) in der Auswertung seiner Juni-Umfrage unter den Mitgliedsbetrieben in NRW fest.",
        target="C’est ce que constate l’Association des employeurs (AGV) dans les résultats qui ressortent d’un sondage effectué en juin auprès des entreprises membres en Rhénanie-du-Nord-Westphalie."
    ),
    Examples(
        source="Das gedruckte Programm wird in den nächsten Tagen unter anderem im Rathaus ausgelegt.",
        target="Le programme imprimé sera disponible à la mairie, entre autres, dans les jours qui viennent."
    ),
    Examples(
        source="Die Konjunkturkurven dieser Indizes für Deutschland und für die Euro-Zone zeigen die Form des Buchstabens \"V\": Auf den steilen Absturz im April folgt ein noch stärkerer Anstieg bis Juli.",
        target="Les courbes de ces indices pour l’Allemagne et pour la zone euro ont la forme de la lettre « V » : À une baisse brutale en avril succède une progression encore plus forte jusque juillet."
    ),
    Examples(
        source="Ein Sprecher von Prevent sagte, das Unternehmen habe keine Kenntnis von den Aufnahmen gehabt.",
        target="Selon un porte-parole de Prevent, l’entreprise n’aurait pas eu connaissance des enregistrements."
    ),
    Examples(
        source="Thyssen-Krupp erwägt offenbar Komplettverkauf der Stahlsparte",
        target="Thyssen-Krupp envisage visiblement de vendre l’intégralité de la branche acier"
    ),
    Examples(
        source="Dank der Landwirtschaft und verwandter Wirtschaftszweige, die den ländlichen Sektor ankurbeln.",
        target="Grâce à l’agriculture et aux secteurs économiques affiliés, qui stimulent le secteur agricole."
    ),
]

TRANSLATION_TEMPLATE = "{} ="


mt_prompts = {'en-de': LLMTask(description=TRANSLATION_DESCR, demonstrations=ENDE_TRANSLATION_EXAMPLES,
                               template=TRANSLATION_TEMPLATE),
             'en-ru': LLMTask(description=TRANSLATION_DESCR, demonstrations=ENRU_TRANSLATION_EXAMPLES,
                                    template=TRANSLATION_TEMPLATE),
             'zh-en': LLMTask(description=TRANSLATION_DESCR, demonstrations=ZHEN_TRANSLATION_EXAMPLES,
                                    template=TRANSLATION_TEMPLATE),
             'nl-en': LLMTask(description=TRANSLATION_DESCR, demonstrations=NLEN_TRANSLATION_EXAMPLES,
                                    template=TRANSLATION_TEMPLATE),
             'is-en': LLMTask(description=TRANSLATION_DESCR, demonstrations=ISEN_TRANSLATION_EXAMPLES,
                                    template=TRANSLATION_TEMPLATE),
             'de-fr': LLMTask(description=TRANSLATION_DESCR, demonstrations=DEFR_TRANSLATION_EXAMPLES,
                                    template=TRANSLATION_TEMPLATE),
              }
