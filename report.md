# Rapport för lösning av car-racing V3 gymnasium miljö

Vi löste gymnasium car racing v3 miljön med två olika metoder, PPO och DQN.
### Metoder

Vi byggde ett prrogram som **sökerparametrar me optuna**, tränar modellerna med parametrarna, evaluerar modellen, göra en visualisering av när tränade agenten löser miljön.

### Processen
Vi hade till början svårigheter att få agenten att lära sig, den ville bara göra "donitser". Senare kom vi fram till att vi hade gjort några små misstag som orsakade ickoptimalt resultat. Medan Det var ganska snabbt att få allt och köra tog det tid att få resultat som vi blev nöjda med.

### Resultat
**Reulsts go here**

#### Tränings grafer

#### Video på agentens körande

### Analys av resultat
Vi kom fram till att PPO gynnade bättre resultat än DQN. Den främsta orsaken till detta är att PPO är en policy-gradient-metod som hanterar kontinuerliga aktionsrum direkt, vilket passar CarRacing-miljön där styrning, gas och broms är kontinuerliga värden. DQN däremot är utformad för diskreta aktionsrum, vilket gör att bilens kontroll blir grov och instabil även med anpassningar.

PPO:s förmåga att uppdatera policyn försiktigt (“proximal updates”) bidrog också till mer stabil inlärning och bättre generalisering över olika banor. DQN tenderade att fastna i lokala optima och visade mer varierande resultat mellan episoder.


