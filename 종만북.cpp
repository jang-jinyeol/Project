#include <iostream>
#include <vector>
#
using namespace std;



bool match(const string& w, const string& s) {
	int pos = 0;

	while (pos < s.size() && pos < w.size() && (w[pos] == '?' || w[pos] == s[pos]))
		pos++;

	if (pos == w.size())
		return pos == s.size();

	if (w[pos] == '*')
		for (int skip = 0; pos + skip <= s.size(); ++skip)
			if (match(w.substr(pos + 1), s.substr(pos + skip)))
				return true;
	return false;
}



int main() {

	//int num,num2;
	//string a;

	//cin >> num;
	//
	//for (int i = 0; i < num; i++) {
	//	cin >> a;
	//	cin >> num2;

	//	for (int j = 0; j < num2; j++) {

	//		cin >> a;
	//	}
	//}

	string a = "*p*";
	string b = "helpp";

	cout<<match(a, b);

	//cout << a.substr(1);

	 
}

