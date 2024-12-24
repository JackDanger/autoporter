const esprima = require('esprima');
const fs = require('fs');

const fileName = process.argv[2];

try {
  const jsContent = fs.readFileSync(fileName, 'utf8');
  const ast = esprima.parseModule(jsContent, { tolerant: true });

  const dependencies = [];
  function traverse(node) {
    if (node &&
        node.type === 'CallExpression' &&
        node.callee.type === 'MemberExpression' &&
        node.callee.property.type === 'Identifier' &&
        node.callee.property.name === 'module') {
          if (node.arguments && node.arguments.length > 0 && node.arguments[0].type === 'Literal') {
            dependencies.push({
                value: node.arguments[0].value,
                loc: node.arguments[0].loc
              });
          }
        }

    for (let key in node) {
      if (node.hasOwnProperty(key) && typeof node[key] === 'object' && node[key] !== null) {
          if (Array.isArray(node[key])) {
            for (let item of node[key]) {
              traverse(item);
            }
          } else {
            traverse(node[key]);
          }
      }
    }
  }
  traverse(ast);

  console.log(JSON.stringify({
      ast: ast,
    dependencies: dependencies
  }));
} catch (e) {
  console.error(e.message);
  process.exit(1);
}
