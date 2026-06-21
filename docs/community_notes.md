
# Nuggets of Gold knowledge from Council of Elders: 

- Some tips: this is a very noisy dataset (maybe 2% signal) with a habit of having streaks (one pattern lasting for a while, and then stopping). So once you have trained a model and it looks good on a decent validation dataset (I use about a 100 weeks for validation), know that your performance when submitting the model can look off for a while (if you happen to catch a streak of market behavior that your model doesn’t excel at.

- People tend to agree that small ensemble models tend to perform best in the tournament. 

--------------------------
By User **svendaj** (Master)
Later data seems to work better with smaller neutralization proportion.
Original Notebooks suggest 50% , but smaller like 25% might work better. 
Test this.

-------------------------
By user: rbot (Grandmaster Council)

train dataset : changes only when version changes (i.e from 5.2 to 5.3)
validation dataset : grows weekly 

------------------------
Forums discussion : 

## How often do you re-train your model?

**anthill**
How sensitive you are to the most recent data may depend on your specific model. Some models do best with up-to-date data, others are looking more for stable signals that don’t change much over time. Models that use more stable signals would ideally have less volatility and so be less susceptible to drawdowns.

But ultimately you can figure out what works best for your model by doing step-forward training. Backtest on the validation data by training on the “freshest” data for a particular validation era, vs. data that is a few weeks old and see which does better.

**spn30n** (Aug 1, 2025)
My model performed well initially, but performance degraded over time. How often should we re-train to incorporate new data?

**anthill** (Aug 1, 2025)
Sensitivity to recent data depends on your model. Some benefit from up-to-date data; others rely on stable signals. Test both approaches on validation data using step-forward training to determine what works best.

**bguberfain** (Grandmaster) (Aug 7, 2025)
I retrain weekly.

**svendaj** (Master)
I retrain weekly, namely because it’s for free. I am fully automated on Kaggle platform, so it costs me nothing and I can focus on experimentation. If there were some costs related, I would certainly think twice about retraining frequency.

The frequency of retraining is almost like trying to time the market - a futile effort. For example, my best public model JOS_KAGGLE_SUNSHINE Profile - Numerai has been trained 10 months ago (no retraining since) and has reasonable 1Y return of 61.7% and its CORR20 performance is at #152 of the models leaderboard.

**jackasspeech2**
You could ensemble the static model with the weekly one to balance long term stability with short term adaptation

--------------------------

**question from new member**
I uploaded a model I am happy with, and staked some NMR on it. 
I wanted to kindly ask about the lifecycle of staked models. 
How soon will I see returns/losses ? Is it instant (i.e next day) , or will I have to wait an era (week) or a month ? 

**reply from lostdev**
when you stake a model slot and a prediction is submitted for a round, the stake is applied to the round. The payout or burn happens when the round closes, based on the CORR and MMC scores for that round. Rounds are about a month long for crypto and classic, and three months for signals.

-------------------------
By user: Andralienware (Team | Grandmaster Council)

Filling in lots of preds with 0's is kind of expected to hurt corr.
Here's how I think about corr--it's the dot product of your predictions (with filling + rank gauss pow 1.5 applied) divided by the product of the standard deviation of the targets and the standard deviation of the (transformed) predictions. You can't control the standard deviation of the target or the target itself, so all you can control is the predictions and (to the extent that inducing ties can change things) the standard deviation of the targets.

Here's a toy example showing why adding zeroes hurt your corr:
Suppose there are only 5 stocks whose predictions you have to submit.
You have a really good model that tells you with 100% confidence that the first stock will have a target value of -2 and the 5th stock will have a target value of 2. The model is also 100% confident that the 3rd stock will have a target value of 0.

Now, even your very good model is unsure about stocks 2 and 4. One of the targets has to be -1 and the other 1, but your model isn't certain about which--it thinks the probability that the second stock will be -1 and the fourth 1 is 75% (and conversely that the second stock will be 1 and the fourth stock will be -1 with 25% probability). 
At first glance the decision is easy: you predict [-2, 0, 0, 0, 2] and lock in all your non-zeros having a positive product with the target.
What's your EV/Expected Corr in this situation?
To make the math easy, let's leave out the term accounting for the target's standard deviation since it's not a function of our choice.
Then the EV boils down to expected dot product divided by the standard deviation of the prediction. We know with certainty that the first target will have a value of -2 and the last 2, so the dot product will be (-2*-2) + (2*2) = 8.
The prediction variance will be the mean of the square deviations from the mean (in this case zero). This will be ((-2)^2 + 0 + 0 + 0 +2^2)/5 = 8/5. This gives us a standard deviation of about 1.264 . 
Consider the alternative of actually predicting for the uncertain stocks and submitting a prediction like the following: [-2, -1, 0, 1, 2].
Now the variance of your prediction is ((-2)^2 + (-1)^2 + 0 + (1)^2 + (2)^2)/5 = 10/5 =2. This means your standard deviation is now sqrt(2) or about 1.415. This means that for predicting -1 and 1 for the second and fourth stocks respectively to be EV maximizing, the new dot product must be at least 1.415/1.264 times as large as it had originally been at 8; i.e. the expected dot product should be at least 8.96 . We already have at least 8 in expected dot product from the certain first and fifth stocks, so we need the two middling predictions to contribute an expected dot product of at least 0.96 . The expected dot product contribution of the two middling stocks for a given probability p of the second stock being -1 is as follows -1 * ((-1) * p + 1  * (1-p)) [this is the ev from predicting -1 for stock 2] + 1 * ((-1) * (1-p) + 1 * p) [ev from predicting 1 for stock 4]. This simplifies to the EV contribution being 4p - 2. In our case p = 0.75, so we get an expected dot product contribution of 4 * 0.75-2 = 3 - 2 = 1. Since 1 is greater than 0.96, we get enough extra dot product from the middling predictions to justify the extra standard deviation of predicting [-2, -1, 0, 1, 2]. 
This toy example shows that since middling predictions contribute so little to the standard devation of the predictions relative to the most extreme predictions, it's hard to justify the loss in dot product from zeroing them out. I think people can get a lot out of working through some of the math on optimizing corr (which is like optimizing dot product subject to an l2 norm constraint on your predictions) and similar metrics for the sake of better understanding the problem at hand. The math for MMC is obviously a bit different, so taking some time to understand that is also helpful. If the math is overly tedious, I think using a chatbot to implement some of these calculations with low dimensional vectors you can visualize can really help with intuition.