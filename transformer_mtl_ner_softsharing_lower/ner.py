import re
import pandas as pd
import nltk
import sys
from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint

import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
from nltk.tag import StanfordNERTagger
nlp = en_core_web_sm.load()


def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent


def parse_document(document):
   document = re.sub('\n', ' ', document)
   if isinstance(document, str):
       document = document
   else:
       raise ValueError('Document is not string!')
   document = document.strip()
   sentences = nltk.sent_tokenize(document)
   sentences = [sentence.strip() for sentence in sentences]
   return sentences

# sample document
text = """
FIFA was founded in 1904 to oversee international competition among the national associations of Belgium, 
Denmark, France, Germany, the Netherlands, Spain, Sweden, and Switzerland. Headquartered in Zürich, its 
membership now comprises 211 national associations. Member countries must each also be members of one of 
the six regional confederations into which the world is divided: Africa, Asia, Europe, North & Central America 
and the Caribbean, Oceania, and South America.
"""

text2 = "marseille , france -lrb- cnn -rrb- the french prosecutor leading an investigation into the crash of germanwings flight 9525 insisted wednesday that he was not aware of any video footage from on board the plane . marseille prosecutor brice robin told cnn that `` so far no videos were used in the crash investigation . '' he added , `` a person who has such a video needs to immediately give it to the investigators . '' robin 's comments follow claims by two magazines , german daily bild and french paris match , of a cell phone video showing the harrowing final seconds from on board germanwings flight 9525 as it crashed into the french alps . all 150 on board were killed . paris match and bild reported that the video was recovered from a phone at the wreckage site . the two publications described the supposed video , but did not post it on their websites . the publications said that they watched the video , which was found by a source close to the investigation . `` one can hear cries of ` my god ' in several languages , '' paris match reported . `` metallic banging can also be heard more than three times , perhaps of the pilot trying to open the cockpit door with a heavy object . towards the end , after a heavy shake , stronger than the others , the screaming intensifies . then nothing . '' `` it is a very disturbing scene , '' said julian reichelt , editor-in-chief of bild online . an official with france 's accident investigation agency , the bea , said the agency is not aware of any such video . lt. col. jean-marc menichini , a french gendarmerie spokesman in charge of communications on rescue efforts around the germanwings crash site , told cnn that the reports were `` completely wrong '' and `` unwarranted . '' cell phones have been collected at the site , he said , but that they `` had n't been exploited yet . '' menichini said he believed the cell phones would need to be sent to the criminal research institute in rosny sous-bois , near paris , in order to be analyzed by specialized technicians working hand-in-hand with investigators . but none of the cell phones found so far have been sent to the institute , menichini said . asked whether staff involved in the search could have leaked a memory card to the media , menichini answered with a categorical `` no . '' reichelt told `` erin burnett : outfront '' that he had watched the video and stood by the report , saying bild and paris match are `` very confident '' that the clip is real . he noted that investigators only revealed they 'd recovered cell phones from the crash site after bild and paris match published their reports . `` that is something we did not know before . ... overall we can say many things of the investigation were n't revealed by the investigation at the beginning , '' he said . what was mental state of germanwings co-pilot ? german airline lufthansa confirmed tuesday that co-pilot andreas lubitz had battled depression years before he took the controls of germanwings flight 9525 , which he 's accused of deliberately crashing last week in the french alps . lubitz told his lufthansa flight training school in 2009 that he had a `` previous episode of severe depression , '' the airline said tuesday . email correspondence between lubitz and the school discovered in an internal investigation , lufthansa said , included medical documents he submitted in connection with resuming his flight training . the announcement indicates that lufthansa , the parent company of germanwings , knew of lubitz 's battle with depression , allowed him to continue training and ultimately put him in the cockpit . lufthansa , whose ceo carsten spohr previously said lubitz was 100 % fit to fly , described its statement tuesday as a `` swift and seamless clarification '' and said it was sharing the information and documents -- including training and medical records -- with public prosecutors . spohr traveled to the crash site wednesday , where recovery teams have been working for the past week to recover human remains and plane debris scattered across a steep mountainside . he saw the crisis center set up in seyne-les-alpes , laid a wreath in the village of le vernet , closer to the crash site , where grieving families have left flowers at a simple stone memorial . menichini told cnn late tuesday that no visible human remains were left at the site but recovery teams would keep searching . french president francois hollande , speaking tuesday , said that it should be possible to identify all the victims using dna analysis by the end of the week , sooner than authorities had previously suggested . in the meantime , the recovery of the victims ' personal belongings will start wednesday , menichini said . among those personal belongings could be more cell phones belonging to the 144 passengers and six crew on board . check out the latest from our correspondents . the details about lubitz 's correspondence with the flight school during his training were among several developments as investigators continued to delve into what caused the crash and lubitz 's possible motive for downing the jet . a lufthansa spokesperson told cnn on tuesday that lubitz had a valid medical certificate , had passed all his examinations and `` held all the licenses required . '' earlier , a spokesman for the prosecutor 's office in dusseldorf , christoph kumpa , said medical records reveal lubitz suffered from suicidal tendencies at some point before his aviation career and underwent psychotherapy before he got his pilot 's license . kumpa emphasized there 's no evidence suggesting lubitz was suicidal or acting aggressively before the crash . investigators are looking into whether lubitz feared his medical condition would cause him to lose his pilot 's license , a european government official briefed on the investigation told cnn on tuesday . while flying was `` a big part of his life , '' the source said , it 's only one theory being considered . another source , a law enforcement official briefed on the investigation , also told cnn that authorities believe the primary motive for lubitz to bring down the plane was that he feared he would not be allowed to fly because of his medical problems ."


def ner_sentences(text):
  # tokenize sentences
  sentences = parse_document(text)
  tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
  print(tokenized_sentences)
  # tag sentences and use nltk's Named Entity Chunker
  tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
  # print('---')
  print(tagged_sentences)
  ne_chunked_sents = [nltk.ne_chunk(tagged) for tagged in tagged_sentences]
  # print('---')
  print(ne_chunked_sents)
  # extract all named entities
  named_entities = []
  for ne_tagged_sentence in ne_chunked_sents:
     for tagged_tree in ne_tagged_sentence:
         # extract only chunks having NE labels
         if hasattr(tagged_tree, 'label'):
             entity_name = ' '.join(c[0] for c in tagged_tree.leaves()) #get NE name
             entity_type = tagged_tree.label() # get NE category
             named_entities.append((entity_name, entity_type))
             # get unique named entities
             named_entities = list(set(named_entities))

  # store named entities in a data frame
  entity_frame = pd.DataFrame(named_entities, columns=['Entity Name', 'Entity Type'])
  # display results
  print(entity_frame)
  return named_entities


if __name__ == '__main__':

  # doc = nlp(text2)
  # pprint([(X.text, X.label_) for X in doc.ents])

  # pprint([(X, X.ent_iob_, X.ent_type_) for X in doc])


  # sent = preprocess(text2)
  # pattern = 'NP: {<DT>?<JJ>*<NN>}'

  # cp = nltk.RegexpParser(pattern)
  # cs = cp.parse(sent)
  # print(cs)


  # iob_tagged = tree2conlltags(cs)
  # pprint(iob_tagged)


  # train_file = open(sys.argv[1], 'r')

  # sentences = train_file.readlines()[:100]
  # ners = []
  # for l in sentences:
  #   named_entities = ner_sentences(l)
  #   ners.append(named_entities)
  # print('====')
  # for x in ners:
  #   print(x)

  # ner_sentences(text2)


  st = StanfordNERTagger('/home/guest/guest24/tim/data/cnn/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz', '/home/guest/guest24/tim/data/cnn/stanford-ner/stanford-ner.jar', encoding = 'utf-8')
  tokenized_text = nltk.word_tokenize(text2)
  tagged = st.tag(tokenized_text)
  print(tagged)
  print('===========')
  for tup in tagged:
    if tup[1] != 'O':
      print(tup)