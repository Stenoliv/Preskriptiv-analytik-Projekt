# Rapport för lösning av car-racing V3 gymnasium miljö

Vi löste gymnasium car racing v3 miljön med två olika metoder, PPO och DQN.
### Metoder

Vi byggde ett prrogram som **sökerparametrar me optuna**, tränar modellerna med parametrarna, evaluerar modellen, göra en visualisering av när tränade agenten löser miljön.

### Processen
Vi hade till början svårigheter att få agenten att lära sig. Det ända result vi kom fram till med både PPO och DQN var att bilen bara körde full gas ut ur banan o gjorde "donuts" där. 
Men till slut hittade vi att vi hade `continuous=True` för PPO som vi tror att hade största inverkan på modelens lärande. Men något annat vi ändra var att vi tränade modelen med `render_mode="rgb_array"`
Vi försökte utnyttja optuna för att hitta bättre parametrar men tiden tog slut i mitten då vi hade svårigheter att få modelen alls att bete sig.
Men i princip är optuna helt fullt fungerande vi bara hann inte testa oss till några optuna variabler. 

### Resultat

#### Tränings grafer

## DQN

![alt text](image-1.png)

#### Video på agentens körande



### Analys av resultat
Vi kom fram till att PPO gynnade bättre resultat än DQN. Den främsta orsaken till detta är att PPO är en policy-gradient-metod som hanterar kontinuerliga aktionsrum direkt, vilket passar CarRacing-miljön där styrning, gas och broms är kontinuerliga värden. DQN däremot är utformad för diskreta aktionsrum, vilket gör att bilens kontroll blir grov och instabil även med anpassningar.

PPO:s förmåga att uppdatera policyn försiktigt (“proximal updates”) bidrog också till mer stabil inlärning och bättre generalisering över olika banor. DQN tenderade att fastna i lokala optima och visade mer varierande resultat mellan episoder.


