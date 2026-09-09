const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../replay_server.py'), 'utf8');
const script = source.match(/<script>([\s\S]*?)<\/script>/)[1].replace('{{ state_json | safe }}', '{}');
new vm.Script(script);
const functions = script.slice(script.indexOf('    function draftConstraintNumber'), script.indexOf('    function collectEditPayload'));
const records = [
  {'Flight Number': 'F123', Price: 100, FlightDate: '2022-03-02'},
  {Name: 'Cafe A', 'Average Cost': 20, City: 'Boston'},
  {NAME: 'Hotel A', price: 90, 'maximum occupancy': 2, city: 'Boston'},
  {Name: 'Duplicate', 'Average Cost': 10, City: 'Boston'},
  {Name: 'Duplicate', 'Average Cost': 30, City: 'Miami'},
];
const summary = {innerHTML: ''};
const context = vm.createContext({
  state: {instances: [{world_state: {reference_information: records}, turns: []}]},
  activeInstanceIndex: 0, activeTurnIndex: 0,
  valueToText: v => typeof v === 'object' ? JSON.stringify(v) : String(v),
  prettyKey: v => v, money: v => '$' + v.toFixed(2), esc: v => String(v),
  textToValue: v => v,
  constraintDraft: [{key:'people_number',valueText:'3'}, {key:'budget',valueText:'1000'}],
  goldActionDraft: {action_payload:{plan:{itinerary:[{day:'2022-03-02', current_city:'Boston', transportation:'F123', breakfast:'Cafe A', accommodation:'Hotel A'}]}}},
  document: {getElementById: () => summary},
  goldActionEditor: {querySelector: () => null},
});
vm.runInContext(functions, context);
vm.runInContext(script.slice(script.indexOf('    function compactSelection'), script.indexOf('    function normalizePriority')), context);
const day = {day:'2022-03-02',current_city:'Boston'};
assert.equal(context.travelSelectionName('transportation', 'Flight F123; departure 10:00; price $100', day), 'F123');
assert.equal(context.travelSelectionName('breakfast', 'Cafe A — average cost $20; rating 4', day), 'Cafe A');
assert.equal(context.travelSelectionName('accommodation', 'Hotel A; included in $180 booking', day), 'Hotel A');
assert.equal(context.travelSelectionName('attraction', 'Park A; Museum B', day), 'Park A; Museum B');
assert.equal(context.travelCostItem('transportation','F123',3,day).subtotal,300);
assert.equal(context.travelCostItem('breakfast','Cafe A',3,day).subtotal,60);
assert.equal(context.travelCostItem('accommodation','Hotel A; included in $180 booking',3,day).subtotal,180);
assert.equal(context.travelCostItem('breakfast','Cafe A; old price $999',3,day).subtotal,60);
assert.equal(context.travelCostItem('breakfast','Unknown restaurant',3,day).unavailable,true);
assert.equal(context.travelCostItem('breakfast','Duplicate',3,{}).unavailable,true);
assert.equal(context.travelCostItem('breakfast','Duplicate',3,day).subtotal,30);
assert.equal(context.extractTravelUnitCost({Name:'No price'}),null);
context.renderTravelCostSummary();
assert.match(summary.innerHTML,/Total trip cost: \$540.00/);
context.goldActionDraft.action_payload.plan.itinerary[0].breakfast = 'Duplicate';
context.renderTravelCostSummary();
assert.match(summary.innerHTML,/Total trip cost: \$510.00/);
context.goldActionDraft.action_payload.plan.itinerary[0].breakfast = 'Unknown';
context.renderTravelCostSummary();
assert.match(summary.innerHTML,/Known costs \(incomplete\)/);
assert.match(summary.innerHTML,/Remaining budget unknown/);
console.log('Travel cost lookup and live totals passed');
