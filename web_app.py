import pandas as pd
import numpy as np
from flask import Flask, request, render_template, Markup
from bet_recommender import BetAssistant


# Initialize app
app = Flask(__name__)

new_matches = []

# ba = BetAssistant()
# ba.update_data()

@app.route('/')
def home():
    ba = BetAssistant()
    ba.update_data()
    return render_template('index.html')

@app.route('/input_matches')
def input_matches():
    ba = BetAssistant()
    ba.update_data()
    match_message = ''
    out_message = ''
    form_html = ''
    team_opp = []
    close_message = ''

    today_datetime = pd.Timestamp.today()
    today_date = pd.to_datetime('{0}-{1}-{2}'.format(today_datetime.year, today_datetime.month, today_datetime.day))

    if ba.eligible_matches == []:
        close_message = 'Currently there are no matches availble to bet on. Please check back after 5pm MTN today.'

    elif len(ba.eligible_matches) > 1:
        form_html = '''
        <form action="/portfolio" method="POST" align="center" style="width:100%">
        '''
        for i, match in enumerate(ba.eligible_matches):
            if pd.to_datetime(match['date']) == today_date:
                day = 'today'
            else:
                day = 'tomorrow'
            team = match['team']
            opponent = match['opponent']
            team_opp.append(team+' vs. '+opponent)
            message = '{0} vs. {1} at {2} MTN {3}<br/>'.format(team, opponent, match['time'], day)
            match_message += message
            form_html += '''
            <div>
                <label>{0} vs. {1} :</label>
             </div>
            <div>
                <label>{0} Odds: </label>
                    <input type='text' name='market_team_odds{2}' width=300px>
            </div>
            <div>
                <label>{1} Odds: </label>
                    <input type='text' name='market_opp_odds{2}' width='300px'>
            </div>
            <br/>
            '''.format(team, opponent, i)
        out_message = 'So far there are {0} matches available to bet on:<br/>'.format(len(ba.eligible_matches))
        close_message += "Please enter in a betting market's odds for one or all of the matches below."
        form_html += '''<br/>
        <div>
            <label>Budget: $</label>
                <input type='text' name='budget' width='300px'>
        </div>
        <div>
            <label>Betting Market Name: </label>
                <input type='text' name='market_name' width='300px'>
        </div>
        <input type="submit" value="Get Portfolio", style="width:50%"/>
        </form>'''

    else:
        match = ba.eligible_matches[0]
        team = match['team']
        opponent = match['opponent']
        if pd.to_datetime(match['date']) == today_date:
            day = 'today'
        else:
            day = 'tomorrow'
        message = '{0} vs. {1} at {2} MTN {3}'.format(team, opponent, match['time'], day)
        match_message += message
        out_message = 'So far there is 1 match available to bet on:'
        close_message = "Please enter in a betting market's odds for this match below."

        form_html = Markup('''<form action="/portfolio" method="POST" align="center" style="width:100%">
        <div>
            <label>{0} vs. {1}</label>
        </div>
        <div>
            <label>{0} Odds: </label>
                <input type='text' name='market_team_odds' width=300px>
        </div>
        <div>
            <label>{1} Odds: </label>
                <input type='text' name='market_opp_odds' width='300px'>
        </div>
        <div>
            <label>Betting Market Name: </label>
                <input type='text' name='market_name' width='300px'>
        </div>
        <input type="submit" value="Get Portfolio", style="width:50%"/>
        </form>'''.format(team, opponent))

    form_html = Markup(form_html)
    match_message = Markup(match_message)
    close_message = Markup(close_message)
    out_message = Markup(out_message)

    return render_template('input_matches.html', match_message=match_message, out_message=out_message, close_message=close_message, team_opp=team_opp, form_html=form_html)

@app.route('/portfolio', methods = ['GET', 'POST'])
def portfolio():
    today_datetime = pd.Timestamp.today()
    today_date = pd.to_datetime('{0}-{1}-{2}'.format(today_datetime.year, today_datetime.month, today_datetime.day))

    team = []
    opponent = []
    market_team_odds = []
    market_opp_odds = []
    odds_list = []
    if len(ba.eligible_matches) > 1:
        for i, match in enumerate(ba.eligible_matches):
            team = match['team']
            opponent = match['opponent']
            market_team_odds = request.form['market_team_odds{0}'.format(i)]
            market_opp_odds = request.form['market_opp_odds{0}'.format(i)]
            if market_opp_odds != '' and market_team_odds != '':
                odds_list.append([match['date'], team, opponent, float(market_team_odds), float(market_opp_odds)])

    elif len(ba.eligible_matches) == 1:
        match = ba.eligible_matches[0]
        team = match['team']
        opponent = match['opponent']
        market_team_odds = request.form['market_team_odds{0}'.format(i)]
        market_opp_odds = request.form['market_opp_odds{0}'.format(i)]
        if market_opp_odds != '' and market_team_odds != '':
            odds_list.append([match['date'], team, opponent, float(market_team_odds), float(market_opp_odds)])

    no_budget = False
    if request.form.get('budget', 'no_budget') == 'no_budget':
        budget = 1
        no_budget = True
    else:
        budget = request.form['budget']

    market_name = request.form['market_name'].lower().replace(' ', '')
    if odds_list != []:
        ba.get_today_portfolio(odds_list, float(budget), market_name, print_out=False)

    message = '''You need to enter in odds data first.
    <form action="/input_matches">
        <input type="submit" value="Enter in the Odds" />
    </form>'''
    if ba.portfolio != []:
        if no_budget:
            if pd.to_datetime(match['match']['date']) == today_date:
                day = 'today'
            else:
                day = 'tomorrow'
            match = ba.portfolio[0]
            message = 'Put money on {0} in the {1} vs {2} match at {3} MTN {4}'.format(match['bet_team'], match['match']['team'], match['match']['opponent'], match['match']['time'], day)
        else:
            message = ''
            for match in ba.portfolio:
                if pd.to_datetime(match['match']['date']) == today_date:
                    day = 'today'
                else:
                    day = 'tomorrow'
                message += 'Put ${0:0.2f} on {1} in the {2} vs {3} match at {4} MTN {5} </br></br>'.format(match['bet_amount'], match['bet_team'], match['match']['team'], match['match']['opponent'], match['match']['time'], day)
    # else:
    #     massage = 'There are no matches to bet on.</br>Check back later after 5pm MTN today.'

    message = Markup(message)

    return render_template('portfolio.html', portfolio_message=message)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
