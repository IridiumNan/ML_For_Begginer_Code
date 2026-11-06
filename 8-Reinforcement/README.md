# Reinforcement Demo

To install requirements, run

```sh
cd 8-Reinforcement/

pip install -r ./requirements.txt
```

That's interesting process for reinforce training.

See output/ dir

This is a graph about success rate.

It increased in beginning and dropped.

After ups and downs several times

It reach 100%
![success_rate](./output/success_rates.png)

This is a graph about average steps to reach `the end`, the end can be wolf (failure) or apple (success).

It means how long will it cost to get a feedback, whatever it is.

And it increase in process.

![avg_steps](./output/total_avg_steps.png)

This is the average success steps which means how many steps enable you to reach the positive feedback. So when it increase, just keep going
![success_steps](./output/success_avg_steps.png)
